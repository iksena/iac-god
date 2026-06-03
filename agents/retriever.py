# agents/retriever.py
from __future__ import annotations

import re

from state import GraphState, RemediationHistory, Message, append_and_cap
from agents.llm_client import _build_client, _call_llm_with_history
from prompts.retriever_prompt import get_query_gen_system
from tools.template_annotator import (
    TemplateAnnotation,
    annotate_template,
    attach_smells,
    render_annotated_template,
    render_annotated_terraform,
    extract_resource_types,
)
from tools.cfn_hybrid_rag import execute_hybrid_retrieval
from tools.retriever_helpers import extract_errors, parse_query_response
from tracking.recorder import ResearchRecorder

# ---------------------------------------------------------------------------
# Line-number detection
# ---------------------------------------------------------------------------

# Format produced by validate_cfn_lint:  "[W3005] line 42 | ..."
_WORD_LINE_RE  = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
# Legacy colon-separated format (yamllint, other tools):  ":115" or ":115:7"
_COLON_LINE_RE = re.compile(r":\d+(:\d+)?")
# Defensive fallback for any remaining raw-dict repr (should no longer appear).
_DICT_LINE_RE  = re.compile(r"'LineNumber'\s*:\s*\d+")

# ---------------------------------------------------------------------------
# Error-resource extraction patterns
# ---------------------------------------------------------------------------

# CFN: cfn-lint error format: "[W3687] | line 109 | Resource: WebServerSecurityGroup | ..."
_CFN_LINT_RESOURCE_RE = re.compile(r"Resource:\s*([\w-]+)", re.IGNORECASE)

# Terraform Pattern 1 — raw terraform-validate HCL error header (multi-line stderr):
#   on main.tf line 12, in resource "aws_vpc" "main":
#   on main.tf line 5,  in data "aws_ami" "ubuntu":
#
# This pattern fires when the raw stderr form is stored directly in
# validation_results (uncommon — most paths store the formatted string).
_TF_VALIDATE_RESOURCE_RE = re.compile(
    r'in\s+(?P<block_type>data|resource)\s+"(?P<type>[a-z][a-z0-9_]*)"\s+"(?P<name>[^"]+)"',
    re.IGNORECASE,
)

# Terraform Pattern 2 — bare address form (raw stderr or tflint output).
#   Managed resource address:  aws_vpc.main           -> type="aws_vpc"
#   Data source address:       data.aws_ami.ubuntu    -> type="data.aws_ami"
_TF_ADDRESS_RE = re.compile(
    r'\b(?P<type>(?:data\.)?(?:aws|google|azurerm|random|local|null|archive|tls)_[a-z][a-z0-9_]*)\.(?P<name>[a-z][a-z0-9_-]*)\b',
    re.IGNORECASE,
)

# Terraform Pattern 3 — tflint bracket notation.
#   Managed resource bracket:  [aws_vpc.main]           -> type="aws_vpc"
#   Data source bracket:       [data.aws_ami.ubuntu]    -> type="data.aws_ami"
_TF_BRACKET_RE = re.compile(
    r'\[(?P<type>(?:data\.)?(?:aws|google|azurerm|random|local|null|archive|tls)_[a-z][a-z0-9_]*)\.(?P<name>[a-z][a-z0-9_-]*)\]',
    re.IGNORECASE,
)

# Terraform Pattern 4 — formatted [ERROR] strings produced by format_tflint_errors().
#
# extract_errors() collects errors from validation_results[*].errors which
# already hold the post-formatted string emitted by format_tflint_errors():
#
#   '[ERROR] | line 287 | Reference to undeclared resource | A data resource
#    "aws_partition" "current" has not been declared in the root module.'
#
#   '[ERROR] | line 353 | Unsupported attribute | This object has no argument,
#    nested block, or exported attribute named "region".'
#
#   '[ERROR] | line 12 | Invalid reference | A managed resource
#    "aws_vpc" "main" has not been declared.'
#
# Patterns 1-3 match raw terraform-validate multi-line stderr; they will
# NEVER match these formatted strings because the 'in resource ...' header
# line is stripped during formatting and only the prose description remains.
#
# This pattern extracts the resource type from two description templates
# terraform-validate uses in its natural-language messages:
#
#   Template A (undeclared / unknown resource):
#     'A data resource "<type>" "<name>" has not been declared'
#     'A managed resource "<type>" "<name>" ...'     (rare but valid)
#
#   Template B (attribute / block errors — no block_type keyword):
#     'resource "<type>" "<name>" does not have attribute'
#     'resource "<type>" "<name>" is ...'            (general)
#
# Group layout:
#   block_type  — 'data' or 'resource' or 'managed' (normalised below)
#   type        — bare resource type string (e.g. 'aws_partition')
#   name        — resource instance name   (e.g. 'current')
#
# The extraction loop below maps:
#   block_type == 'data'           -> rtype = 'data.<type>'
#   block_type == 'resource'|'managed' -> rtype = '<type>'
_TF_FORMATTED_ERROR_RE = re.compile(
    r'(?:A\s+)?(?P<block_type>data|managed|resource)\s+resource\s+'
    r'"(?P<type>[a-z][a-z0-9_]*)"\s+"(?P<name>[^"]+)"',
    re.IGNORECASE,
)


def _errors_have_line_numbers(errors: list[str]) -> bool:
    """Return True if at least one error string contains a line number reference.

    Checks three formats in priority order:
      1. Word form:   'line 42'          (cfn-lint / tflint / terraform-validate)
      2. Colon form:  ':115' or ':115:7' (yamllint and other validators)
      3. Dict repr:   "{'LineNumber': 42}" (legacy fallback, should not appear)
    """
    for e in errors:
        if _WORD_LINE_RE.search(e) or _COLON_LINE_RE.search(e) or _DICT_LINE_RE.search(e):
            return True
    return False


def _extract_error_resources(
    errors: list[str],
    annotation: TemplateAnnotation | None,
    deploy_validation_result: dict | None,
) -> set[str]:
    """Extract resource type names for resources that have active errors.

    Handles two distinct ID schemes depending on iac_type:

    CloudFormation (logical ID -> AWS type):
      - cfn-lint errors:      "Resource: <LogicalId>"
      - deploy failed_resources: [{"logical_name": "...", "status_reason": "..."}]
      Logical IDs are mapped to AWS types via the annotation
      (e.g. "WebServerSG" -> "AWS::EC2::SecurityGroup").

    Terraform (resource address -> namespaced provider type):
      Four patterns cover the full range of error string formats:

      Pattern 1 (_TF_VALIDATE_RESOURCE_RE) — raw terraform-validate stderr:
        'in resource "aws_vpc" "main"'         -> rtype = "aws_vpc"
        'in data "aws_ami" "ubuntu"'           -> rtype = "data.aws_ami"
        Fires only when raw multi-line stderr is stored verbatim in state
        (uncommon — most validators pre-format before writing to state).

      Pattern 2 (_TF_ADDRESS_RE) — bare Terraform address:
        'aws_vpc.main'           -> type="aws_vpc"
        'data.aws_ami.ubuntu'    -> type="data.aws_ami"
        Common in tflint output and deploy failed_resources.

      Pattern 3 (_TF_BRACKET_RE) — tflint bracket notation:
        '[aws_vpc.main]'         -> type="aws_vpc"
        '[data.aws_ami.ubuntu]'  -> type="data.aws_ami"

      Pattern 4 (_TF_FORMATTED_ERROR_RE) — formatted [ERROR] strings:
        extract_errors() collects from validation_results[*].errors which
        store the post-formatted string produced by format_tflint_errors():

          '[ERROR] | line 287 | Reference to undeclared resource | A data
           resource "aws_partition" "current" has not been declared in the
           root module.'

        Patterns 1-3 will NEVER match this form because the raw
        'in data "aws_partition" "current"' header line is consumed during
        formatting and only the prose description survives.  Pattern 4
        extracts the type from the natural-language description:

          'A data resource "aws_partition" "current" has not been declared'
          -> block_type="data", type="aws_partition" -> rtype="data.aws_partition"

          'resource "aws_vpc" "main" does not have attribute "region"'
          -> block_type="resource", type="aws_vpc" -> rtype="aws_vpc"

    known_tf_types construction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    template_annotator._parse_terraform() sets:
      resource_id  = "data.{rtype}.{rname}"  for data blocks
      resource_id  = "{rtype}.{rname}"       for managed resources
      resource_type = bare type (no prefix)  in both cases

    The loop below reads resource_id to detect data blocks and adds the
    "data." prefix to resource_type before inserting into known_tf_types,
    producing a set whose members are directly comparable to the rtype
    strings assembled by the four patterns above.

    Returns an empty set when annotation is unavailable or no resource
    identifiers are found, which causes execute_*_retrieval() to fall back to
    fetching schema for all seed resources.
    """
    if not annotation:
        return set()

    error_resource_types: set[str] = set()

    # -----------------------------------------------------------------
    # CloudFormation path: resolve logical IDs -> AWS resource types
    # -----------------------------------------------------------------
    logical_to_type: dict[str, str] = {
        r.resource_id: r.resource_type
        for r in annotation.resources
        if r.resource_id and r.resource_type
    }

    error_logical_ids: set[str] = set()

    for error_str in errors:
        m = _CFN_LINT_RESOURCE_RE.search(error_str)
        if m:
            error_logical_ids.add(m.group(1).strip())

    if deploy_validation_result and not deploy_validation_result.get("passed"):
        for fr in deploy_validation_result.get("failed_resources", []):
            logical_id = fr.get("logical_name") or ""
            if logical_id:
                error_logical_ids.add(logical_id.strip())

    for logical_id in error_logical_ids:
        resource_type = logical_to_type.get(logical_id)
        if resource_type:
            error_resource_types.add(resource_type)

    # -----------------------------------------------------------------
    # Terraform path: extract namespaced resource type from error address
    # -----------------------------------------------------------------
    # Build known_tf_types with explicit namespace prefixing.
    #
    # template_annotator stores:
    #   resource_id  = "data.{rtype}.{rname}"  for data blocks
    #   resource_id  = "{rtype}.{rname}"        for managed resources
    #   resource_type = bare type (no prefix)   in both cases
    #
    # We detect data blocks via the "data." prefix on resource_id and add
    # the same prefix to resource_type so known_tf_types mirrors the
    # "data." convention used in tf_knowledge_graph.json.
    known_tf_types: set[str] = set()
    for r in annotation.resources:
        if not r.resource_type:
            continue
        if r.resource_id and r.resource_id.startswith("data."):
            known_tf_types.add(f"data.{r.resource_type}")
        else:
            known_tf_types.add(r.resource_type)

    if known_tf_types and not any(t.startswith("AWS::") for t in known_tf_types):
        for error_str in errors:
            # ----------------------------------------------------------
            # Pattern 1: raw 'in resource "aws_vpc" "main"' form
            #            or 'in data "aws_ami" "ubuntu"'
            # Fires when raw terraform-validate stderr is stored in state.
            # ----------------------------------------------------------
            for m in _TF_VALIDATE_RESOURCE_RE.finditer(error_str):
                rtype = m.group("type").lower()
                if m.group("block_type").lower() == "data":
                    rtype = f"data.{rtype}"
                if rtype in known_tf_types:
                    error_resource_types.add(rtype)

            # ----------------------------------------------------------
            # Pattern 2: bare address  aws_vpc.main / data.aws_ami.ubuntu
            # (?:data\.)? is inside the capture group, prefix preserved.
            # ----------------------------------------------------------
            for m in _TF_ADDRESS_RE.finditer(error_str):
                rtype = m.group("type").lower()
                if rtype in known_tf_types:
                    error_resource_types.add(rtype)

            # ----------------------------------------------------------
            # Pattern 3: bracket  [aws_vpc.main] / [data.aws_ami.ubuntu]
            # ----------------------------------------------------------
            for m in _TF_BRACKET_RE.finditer(error_str):
                rtype = m.group("type").lower()
                if rtype in known_tf_types:
                    error_resource_types.add(rtype)

            # ----------------------------------------------------------
            # Pattern 4: formatted [ERROR] string from format_tflint_errors()
            #
            # extract_errors() feeds errors from validation_results[*].errors
            # which are already the formatted single-line strings:
            #
            #   '[ERROR] | line 287 | Reference to undeclared resource |
            #    A data resource "aws_partition" "current" has not been
            #    declared in the root module.'
            #
            # Patterns 1-3 never match this form.  This pattern extracts
            # the resource type from the prose description embedded in the
            # last pipe-delimited field.
            #
            # block_type normalisation:
            #   'data'              -> prefix with 'data.'
            #   'resource'/'managed' -> no prefix (managed resource)
            # ----------------------------------------------------------
            for m in _TF_FORMATTED_ERROR_RE.finditer(error_str):
                rtype = m.group("type").lower()
                block_type = m.group("block_type").lower()
                if block_type == "data":
                    rtype = f"data.{rtype}"
                # 'resource' and 'managed' both mean a managed resource
                # — no prefix needed.
                if rtype in known_tf_types:
                    error_resource_types.add(rtype)

    if error_resource_types:
        print(
            f"[Retriever] Error resources scoped to: {sorted(error_resource_types)}"
        )
    else:
        print("[Retriever] No error resources resolved — Neo4j will use full seed set.")

    return error_resource_types


def _annotate_safely(
    template: str,
    smell_report: list[dict] | None,
    iac_type: str = "cloudformation",
) -> TemplateAnnotation | None:
    """Parse and annotate the template for resource-type seeding.

    Returns None on parse failure so callers degrade gracefully.
    Only used to extract resource types for Neo4j seeding — rendering
    is now done directly against the raw template string.

    Passes a synthetic file_path hint ('<in-memory>.tf' for Terraform) so
    that annotate_template() picks the correct parser without re-detection.
    """
    if not template:
        return None
    try:
        file_path_hint = "<in-memory>.tf" if iac_type == "terraform" else "<in-memory>.yaml"
        annotation = annotate_template(file_path=file_path_hint, content=template)
        if smell_report:
            annotation = attach_smells(annotation, smell_report)
        print(f"[Retriever] Annotation: {len(annotation.resources)} resources parsed.")
        return annotation
    except Exception as exc:
        print(f"[Retriever] Annotation failed (non-fatal): {exc}")
        return None


# ---------------------------------------------------------------------------
# Annotated template builder — iac_type-aware
# ---------------------------------------------------------------------------

def _build_retrieval_annotated_template(
    template: str,
    errors: list[str],
    stage_errors: dict[str, list[str]],
    iac_type: str,
) -> str:
    """Render the template with inline error annotations for the retrieval prompt.

    Mirrors the remediator pattern:

    CloudFormation:
        render_annotated_template(yaml, flat_errors) — uses the existing flat
        error list; the renderer extracts line numbers and also builds a
        header block for lineless errors.

    Terraform:
        render_annotated_terraform(hcl, stage_errors) — only tflint /
        terraform-validate errors are annotated inline (the only stages that
        embed HCL line numbers).  Security and deploy errors are excluded
        because they carry no line reference and are already surfaced in
        the validation_errors section of the prompt.
    """
    if iac_type == "terraform":
        return render_annotated_terraform(
            template_hcl=template,
            stage_errors=stage_errors,
        )
    return render_annotated_template(
        template_yaml=template,
        errors=errors,
    )


def build_retrieval_prompt(
    errors: list[str],
    template: str | None,
    annotation: TemplateAnnotation | None,
    remediation_history: list[RemediationHistory],
    iac_type: str = "cloudformation",
    stage_errors: dict[str, list[str]] | None = None,
) -> str:
    """Assemble the single user-turn message for the query-generation LLM call.

    Sections (in order):
      1. Validation errors — rich format: [RuleId] line N | Resource: X | message | description
      2. Full template with inline ERROR comments at the exact reported lines
         (when errors carry line numbers), or plain template fallback.
         Annotation is iac_type-aware: CFN uses render_annotated_template;
         Terraform uses render_annotated_terraform (tflint/terraform-validate
         only) via _build_retrieval_annotated_template.
      3. Prior retrieval-query history to avoid duplicate lookups.

    Pure function — no I/O, no LLM calls, fully unit-testable.
    Only structural errors (cfn-lint / tflint / terraform-validate / deploy) are
    included here; security errors are routed directly to
    execute_security_retrieval() without going through the LLM.
    """
    parts: list[str] = [
        "## Validation Errors\n" + "\n".join(f"- {e}" for e in errors)
    ]

    if template and _errors_have_line_numbers(errors):
        annotated = _build_retrieval_annotated_template(
            template=template,
            errors=errors,
            stage_errors=stage_errors or {},
            iac_type=iac_type,
        )
        parts.append(
            "## IaC Template (errors annotated at reported lines)\n"
            "Lines prefixed with `# ERROR:` mark the exact location the validator\n"
            "reported a violation. Use the resource name and rule description\n"
            "from the error list above as the primary signal for schema retrieval.\n"
            f"```\n{annotated}\n```"
        )
    elif template:
        parts.append(
            "## IaC Template\n"
            "No line-number annotations available — use the error messages\n"
            "above to identify which resource types and attributes need schema retrieval.\n"
            f"```\n{template}\n```"
        )

    history_lines: list[str] = []
    for entry in remediation_history:
        if not entry.get("retrieval_queries"):
            continue
        queries_str = "\n".join(f"  - {q}" for q in entry["retrieval_queries"])
        history_lines.append(f"### Iteration {entry['iteration']} queries used:\n{queries_str}")

    if history_lines:
        parts.append(
            "## Prior Retrieval Queries\n"
            "These queries were already used in previous iterations.\n"
            "Generate DIFFERENT queries targeting unexplored resource.attribute combinations.\n"
            "\n" + "\n\n".join(history_lines)
        )

    return "\n\n".join(parts)


def _get_active_error_types(state: GraphState) -> tuple[bool, bool]:
    """Determine if active errors require schema context, security context, or both.

    Schema retrieval is triggered by any structural stage failure:
      CloudFormation: "yaml" or "cfn-lint" stage
      Terraform:      "tflint" or "terraform-validate" stage
      Both:           live deployment failures

    Note: tflint (Terraform Stage 1) and cfn-lint (CFN Stage 2) are symmetric
    structural linters — both trigger schema RAG so the repair loop receives
    provider/resource schema context regardless of which structural stage
    caught the error. This parity is required for the generalisation hypothesis.

    Security retrieval is triggered by checkov / trivy stage failures.
    """
    has_schema = False
    has_security = False

    latest_by_stage: dict[str, dict] = {}
    for result in state.get("validation_results", []):
        stage = str(result.get("stage") or "").strip()
        if stage:
            latest_by_stage[stage] = result

    if not latest_by_stage.get("yaml", {}).get("passed", True):
        has_schema = True
    if not latest_by_stage.get("cfn-lint", {}).get("passed", True):
        has_schema = True
    if not latest_by_stage.get("tflint", {}).get("passed", True):
        has_schema = True
    if not latest_by_stage.get("terraform-validate", {}).get("passed", True):
        has_schema = True

    deploy = state.get("deploy_validation_result")
    if deploy and not deploy.get("passed") and deploy.get("target") != "skipped":
        has_schema = True

    if not latest_by_stage.get("checkov", {}).get("passed", True):
        has_security = True
    if not latest_by_stage.get("trivy", {}).get("passed", True):
        has_security = True

    if not has_schema and not has_security:
        has_schema, has_security = True, True

    return has_schema, has_security


def _call_query_generator(
    user_content: str,
    system_prompt: str,
) -> tuple[str, str, dict | None]:
    """Send the retrieval prompt to the LLM without conversation history.

    The retriever is intentionally stateless across iterations: all context
    needed to generate non-redundant queries is already embedded in
    user_content via the '## Prior Retrieval Queries' section built from
    remediation_history.  Carrying a rolling LLM conversation history added
    no value and risked stale prior-iteration reasoning leaking into the
    current query set.

    Returns (model, raw_response, token_usage).
    """
    client, model = _build_client()
    raw_response, usage = _call_llm_with_history(
        client,
        model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return model, raw_response, usage


def _build_tf_seed_resources(annotation: TemplateAnnotation | None) -> set[str]:
    """Build the seed_resources set for Terraform RAG with correct data. prefixing.

    extract_resource_types() returns bare resource_type strings for all
    resources — it has no knowledge of the data. prefix convention used in
    the TF knowledge graph (tf_knowledge_graph.json, Neo4j TFResource nodes).

    This function mirrors the prefixing logic in known_tf_types inside
    _extract_error_resources() so both seed_resources (full-template fallback)
    and known_tf_types (error-scoped lookups) use the same namespace keys.

    Canonical mapping (from template_annotator._parse_terraform):
      resource_id  = "data.{rtype}.{rname}"  for data blocks
      resource_id  = "{rtype}.{rname}"        for managed resources
      resource_type = bare type (no prefix)   in both cases

    Result:
      data "aws_vpc" "default"    -> seed entry: "data.aws_vpc"
      resource "aws_s3_bucket" "x" -> seed entry: "aws_s3_bucket"
    """
    if not annotation:
        return set()
    seed: set[str] = set()
    for r in annotation.resources:
        if not r.resource_type:
            continue
        if r.resource_id and r.resource_id.startswith("data."):
            seed.add(f"data.{r.resource_type}")
        else:
            seed.add(r.resource_type)
    return seed


def retriever_agent(state: GraphState, recorder: ResearchRecorder) -> GraphState:
    """Dedicated retrieval agent.

    Orchestration steps:
      1. Extract structural + deploy errors (security stages excluded).
      2. Build stage_errors dict from validation_results for iac_type-aware
         template annotation (mirrors remediator pattern).
      3. Annotate the template to seed resource types for the RAG layer.
         For Terraform, _parse_terraform includes both resource AND data
         blocks in the unified resources list, with resource_id formatted as
         "data.{rtype}.{rname}" for data blocks and "{rtype}.{rname}" for
         managed resources.
      4. Build the retrieval prompt (pure, no I/O) using iac_type-aware
         section headers and system prompt via get_query_gen_system().
      5. When has_schema:
           - CloudFormation: call LLM to generate schema_queries, then run
             ChromaDB + Neo4j hybrid retrieval via execute_hybrid_retrieval().
           - Terraform: call LLM to generate schema_queries, then run
             execute_terraform_retrieval() (stub today; wired for TF corpus).
      6. When has_security: extract AVD/Trivy IDs from raw errors via regex,
         then run a direct Neo4j graph lookup — NO LLM query generation for
         security (IDs are already explicit in the validator output).
         iac_type is forwarded so Neo4j queries the correct framework nodes
         (frameworks=['terraform','tf'] vs ['cfn','cloudformation']).
      7. Persist prompt, response, queries, and full context to
         retriever_history.txt via the recorder.
      8. Return retriever_context, retriever_queries, and updated
         retriever_history into state.
    """
    iteration = state["current_iteration"]
    iac_type = state.get("iac_type", "cloudformation")
    print(f"\n[Retriever] Building context (iteration {iteration}, iac_type={iac_type})...")

    deploy_validation_result = state.get("deploy_validation_result")

    errors = extract_errors(
        state.get("validation_results", []),
        deploy_validation_result,
    )

    stage_errors: dict[str, list[str]] = {}
    for result in state.get("validation_results", []):
        stage = str(result.get("stage") or "").strip()
        if not stage:
            continue
        errs = [str(e) for e in result.get("errors", []) if str(e).strip()]
        if errs:
            stage_errors[stage] = errs

    template = state.get("iac_template", "")
    annotation = _annotate_safely(
        template=template,
        smell_report=state.get("smell_report"),
        iac_type=iac_type,
    )

    has_schema, has_security = _get_active_error_types(state)
    print(f"[Retriever] Routing -> Schema: {has_schema} | Security: {has_security}")

    schema_queries: list[str] = []
    iac_context = ""
    llm_record = None

    if has_schema:
        has_line_numbers = _errors_have_line_numbers(errors)
        print(
            f"[Retriever] {'Line numbers detected — annotated template' : <45} "
            f"{'included' if has_line_numbers else 'NOT included (plain template used)'}."
        )

        query_gen_system = get_query_gen_system(iac_type)

        user_content = build_retrieval_prompt(
            errors=errors,
            template=template,
            annotation=annotation,
            remediation_history=state.get("remediation_history", []),
            iac_type=iac_type,
            stage_errors=stage_errors,
        )

        model, raw_response, usage = _call_query_generator(
            user_content=user_content,
            system_prompt=query_gen_system,
        )
        parsed_queries = parse_query_response(raw_response)
        schema_queries = parsed_queries.get("schema_queries", [])

        if not schema_queries:
            schema_queries = errors[:8]

        llm_record = recorder.record_llm_call(
            state=state,
            agent="retriever",
            model=model,
            prompt=f"SYSTEM:\n{query_gen_system}\n\nUSER:\n{user_content}",
            response=raw_response,
            token_usage=usage,
        )

        # -----------------------------------------------------------------
        # Build seed_resources
        # -----------------------------------------------------------------
        # For CloudFormation: use extract_resource_types() directly — it
        # returns AWS:: type strings which are the correct Neo4j node keys.
        #
        # For Terraform: use _build_tf_seed_resources() which applies the
        # same "data." prefix logic used in known_tf_types.  This is the
        # fix for Bug 1 — data sources were previously passed as bare type
        # strings (e.g. "aws_availability_zones") but Neo4j TFResource
        # nodes are keyed as "data.aws_availability_zones" after the
        # 03_parse_and_merge_tf.py Phase 1 fix.
        #
        # Both paths must produce sets whose members directly match the
        # TFResource / Resource node names in Neo4j.
        if iac_type == "terraform":
            seed_resources = _build_tf_seed_resources(annotation)
        else:
            seed_resources = extract_resource_types(annotation)

        # Diagnostic: log seed_resources so missing providers (e.g. random_id)
        # are immediately visible in the run output without a debugger.
        print(f"[Retriever] seed_resources ({len(seed_resources)}): {sorted(seed_resources)}")

        error_resources = _extract_error_resources(
            errors=errors,
            annotation=annotation,
            deploy_validation_result=deploy_validation_result,
        ) or None

        if iac_type == "terraform":
            from tools.tf_hybrid_rag import execute_terraform_retrieval
            iac_context = execute_terraform_retrieval(
                retrieval_queries=schema_queries,
                seed_resources=seed_resources,
                error_resources=error_resources,
            )
        else:
            iac_context = execute_hybrid_retrieval(
                retrieval_queries=schema_queries,
                seed_resources=seed_resources,
                error_resources=error_resources,
            )
    else:
        user_content = ""
        raw_response = ""
        model = ""
        usage = None

    security_context = ""
    security_ids: list[str] = []

    if has_security:
        from tools.security_hybrid_rag import execute_security_retrieval, extract_trivy_check_ids
        security_ids = extract_trivy_check_ids(errors)
        security_context = execute_security_retrieval(
            raw_errors=errors,
            iac_type=iac_type,
        )

    unified_context = "\n\n".join(
        part for part in (iac_context, security_context) if part.strip()
    )

    retrieval_queries = schema_queries + security_ids

    query_gen_system = get_query_gen_system(iac_type)

    user_msg: Message = {"role": "user", "content": user_content if has_schema else ""}
    assistant_msg: Message = {"role": "assistant", "content": raw_response if has_schema else ""}

    recorder.append_retriever_history_entry(
        iteration=iteration,
        prompt=f"SYSTEM:\n{query_gen_system}\n\nUSER:\n{user_content if has_schema else ''}",
        response=raw_response if has_schema else "",
        retrieval_queries=retrieval_queries,
        context_chars=len(unified_context),
        retrieved_context=unified_context,
    )

    print(
        f"[Retriever] Context: {len(unified_context)} char(s) | "
        f"{len(schema_queries)} schema quer(y/ies) | "
        f"{len(security_ids)} security ID(s) resolved."
    )

    return {
        "retriever_context":  unified_context,
        "retriever_queries":  retrieval_queries,
        "llm_call_log":       state["llm_call_log"] + ([llm_record] if llm_record else []),
        "retriever_history":  append_and_cap(
            state.get("retriever_history", []), user_msg, assistant_msg
        ),
    }
