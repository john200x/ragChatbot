import json
import os
import boto3
from botocore.exceptions import ClientError
import httpx  # for AI Guard HTTP call

# --- AWS Bedrock config ---

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")

KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
MODEL_ARN = os.environ["MODEL_ARN"]

# --- AI Guard (Zscaler Eclipse) config ---

AIGD_API_KEY = os.environ["ZAG_API_KEY"]
AIGD_URL = "https://api.zseclipse.net/v1/detection/execute-policy"
AIGD_POLICY_ID = 1154

# ---------------------------------------------------------------------------
# AI Guard detector -> human-friendly message map
# ---------------------------------------------------------------------------

DETECTOR_LABELS = {
    "pii":           "Personal Identifiable Information (PII) detected",
    "toxicity":      "Toxic or harmful content detected",
    "injection":     "Prompt injection attempt detected",
    "secrets":       "Sensitive secrets or credentials detected",
    "gibberish":     "Incoherent or gibberish content detected",
    "malicious_url": "Malicious URL detected",
    "data_leakage":  "Potential data leakage detected",
}


def friendly_reason(detectors: list) -> str:
    """Map a list of raw detector names to human-friendly descriptions.

    Unknown detector names fall back to 'Policy violation: <name>'.
    Returns a ' | ' separated string, or a default message if the list is empty.
    """
    if not detectors:
        return "Policy violation (unspecified)"
    reasons = [
        DETECTOR_LABELS.get(str(d).lower(), f"Policy violation: {d}")
        for d in detectors
    ]
    return " | ".join(reasons)


def extract_blocking_detectors(aiguard_result: dict) -> list:
    """Extract the list of triggered detector names from any AI Guard response shape.

    Handles three known response shapes:
      Shape A - native execute-policy:  detectorResponses.<name>.action == "BLOCK"
      Shape B - OpenAI-proxy style:     blockingDetectors: [...]
      Shape C - nested details:         details.blockingDetectors: [...]
    """
    detectors = []

    # Shape A: iterate detectorResponses dict; collect keys whose action is BLOCK
    detector_responses = aiguard_result.get("detectorResponses", {})
    if detector_responses:
        for name, info in detector_responses.items():
            if isinstance(info, dict):
                det_action = (info.get("action") or "").upper()
                if det_action == "BLOCK":
                    detectors.append(name.lower())

    # Shape B: top-level blockingDetectors list (OpenAI-proxy)
    if not detectors and "blockingDetectors" in aiguard_result:
        raw = aiguard_result["blockingDetectors"]
        if isinstance(raw, list):
            detectors = [str(d).lower() for d in raw]

    # Shape C: nested under 'details'
    if not detectors:
        nested = aiguard_result.get("details", {}).get("blockingDetectors", [])
        if isinstance(nested, list):
            detectors = [str(d).lower() for d in nested]

    return detectors


def build_block_message(aiguard_result: dict, direction: str = "request") -> str:
    """Build a complete user-facing block message with friendly reason labels
    and a transaction reference ID.

    Args:
        aiguard_result: The raw AI Guard API response dict.
        direction: 'request' for inbound (prompt) blocks,
                   'response' for outbound (AI answer) blocks.
    """
    detectors = extract_blocking_detectors(aiguard_result)

    txn_id = (
        aiguard_result.get("transactionId")
        or aiguard_result.get("transaction_id")
        or "N/A"
    )

    reason_str = friendly_reason(detectors)

    if direction == "response":
        return (
            f"\u26a0\ufe0f The AI response was blocked by AI Security Policy.\n"
            f"Reason(s): {reason_str}\n"
            f"Reference ID: {txn_id}"
        )
    else:
        return (
            f"\u26a0\ufe0f Your request was blocked by AI Security Policy.\n"
            f"Reason(s): {reason_str}\n"
            f"Reference ID: {txn_id}"
        )


def build_system_prompt():
    return (
        "You are a helpful assistant that answers employee questions "
        "using the provided context from the knowledge base. "
        "If the answer is not in the documents, say you don't know."
    )


def call_aiguard(prompt: str) -> dict:
    """
    Call Zscaler Eclipse AI Guard execute-policy endpoint for INBOUND content
    (direction=IN). Checks the user prompt BEFORE sending to Bedrock.

    Returns:
        dict with at minimum an 'action' key: 'allow' or 'block'
    """
    headers = {
        "Authorization": f"Bearer {AIGD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "policyId": AIGD_POLICY_ID,
        "direction": "IN",
        "content": prompt,
    }

    resp = httpx.request("POST", AIGD_URL, headers=headers, json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def call_aiguard_response(answer: str) -> dict:
    """
    Call Zscaler Eclipse AI Guard execute-policy endpoint for OUTBOUND content
    (direction=OUT). Checks the AI-generated answer BEFORE returning to the user.

    This catches cases where the AI response itself may contain policy violations
    such as data leakage, PII in generated text, or toxic AI output.

    Returns:
        dict with at minimum an 'action' key: 'allow' or 'block'
    """
    headers = {
        "Authorization": f"Bearer {AIGD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "policyId": AIGD_POLICY_ID,
        "direction": "OUT",
        "content": answer,
    }

    resp = httpx.request("POST", AIGD_URL, headers=headers, json=payload, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def lambda_handler(event, context):
    # Handle CORS preflight
    method = (
        event.get("httpMethod")
        or event.get("requestContext", {}).get("http", {}).get("method")
    )

    if method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": _cors_headers(),
            "body": "",
        }

    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body or "{}")

        query = body.get("query") or body.get("question")
        session_id = body.get("sessionId")

        user_prompt = query

        if not query:
            return _response(
                400,
                {"error": "query (or question) field is required"},
            )

        # Build bedrock prompt (system + user question)
        bedrock_prompt = f"{build_system_prompt()}\n\nQuestion:\n{query}\n\nAnswer:"

        # \u2500\u2500 STEP 1: AI Guard INBOUND check (prompt) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        try:
            aiguard_inbound = call_aiguard(user_prompt)
        except Exception as e:
            # Fail-closed: reject the request if AI Guard is unreachable
            return _response(
                502,
                {
                    "error": "AI Guard inbound check failed",
                    "message": str(e),
                },
            )

        inbound_action = (aiguard_inbound.get("action") or "").upper()
        if inbound_action != "ALLOW":
            # Build a human-friendly block message with per-detector labels
            block_message = build_block_message(aiguard_inbound, direction="request")
            return _response(
                403,
                {
                    "answer": block_message,         # shown to the end-user
                    "blocked_at": "inbound",          # where in the pipeline it was blocked
                    "aiguard": aiguard_inbound,       # raw result for audit/admin
                },
            )

        # \u2500\u2500 STEP 2: Bedrock retrieve_and_generate \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        request_params = {
            "input": {"text": bedrock_prompt},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5,
                        }
                    },
                },
            },
        }

        if session_id:
            request_params["sessionId"] = session_id

        resp = bedrock_agent_runtime.retrieve_and_generate(**request_params)

        answer = resp["output"]["text"]
        citations = resp.get("citations", [])
        new_session_id = resp.get("sessionId")

        # \u2500\u2500 STEP 3: AI Guard OUTBOUND check (AI response) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        try:
            aiguard_outbound = call_aiguard_response(answer)
        except Exception as e:
            # Fail-closed: withhold the answer if AI Guard is unreachable
            return _response(
                502,
                {
                    "error": "AI Guard outbound check failed",
                    "message": str(e),
                },
            )

        outbound_action = (aiguard_outbound.get("action") or "").upper()
        if outbound_action != "ALLOW":
            # The AI-generated answer itself triggered a policy violation
            block_message = build_block_message(aiguard_outbound, direction="response")
            return _response(
                403,
                {
                    "answer": block_message,          # shown to the end-user
                    "blocked_at": "outbound",          # where in the pipeline it was blocked
                    "aiguard": aiguard_outbound,       # raw result for audit/admin
                },
            )

        # \u2500\u2500 STEP 4: All checks passed \u2014 return answer to user \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        return _response(
            200,
            {
                "answer": answer,
                "sessionId": new_session_id,
                "citations": citations,
                "aiguard_inbound": aiguard_inbound,   # inbound check result (audit)
                "aiguard_outbound": aiguard_outbound, # outbound check result (audit)
            },
        )

    except ClientError as e:
        return _response(
            500,
            {
                "error": "AWS error",
                "code": e.response["Error"]["Code"],
                "message": e.response["Error"]["Message"],
            },
        )

    except Exception as e:
        return _response(
            500,
            {
                "error": "Internal server error",
                "message": str(e),
            },
        )


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": (
            "Content-Type,Authorization,X-Amz-Date,X-Api-Key,"
            "X-Amz-Security-Token"
        ),
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Access-Control-Max-Age": "3600",
    }


def _response(status_code, body_dict):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            **_cors_headers(),
        },
        "body": json.dumps(body_dict, indent=2),
    }
