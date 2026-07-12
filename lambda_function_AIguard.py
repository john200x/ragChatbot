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

def build_system_prompt():
    return (
        "You are a helpful assistant that answers employee questions "
        "using the provided context from the knowledge base. "
        "If the answer is not in the documents, say you don't know."
    )


def call_aiguard(prompt: str) -> dict:
    """
    Call Zscaler Eclipse AI Guard execute-policy endpoint for inbound content.
    Assumes response contains fields:
      - action: 'allow' or 'block'
      - violations: list of objects with 'reason' etc. (if any)
    """
    headers = {
        "Authorization": f"Bearer {AIGD_API_KEY}",
        "Content-Type": "application/json",
    }
    #prompt_string = "tell me a dog joke"
    payload = {
        "policyId": AIGD_POLICY_ID,
        "direction": "IN",
        "content": prompt,
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

        # Build user prompt for Bedrock
        bedrock_prompt = f"{build_system_prompt()}\n\nQuestion:\n{query}\n\nAnswer:"

        # 1) Call AI Guard before Bedrock
        try:
            aiguard_result = call_aiguard(user_prompt)
        except Exception as e:
            # Fail-closed or fail-open is your policy decision; here: fail-closed
            return _response(
                502,
                {
                    "error": "AI Guard call failed",
                    "message": str(e),
                },
            )

        action = aiguard_result.get("action")
        #if action.upper() != "ALLOW":
        #    # Block if AI Guard says block
        #    return _response(
        #       403,
        #        {
        #            "error": "Request blocked by AI Guard policy",
        #            "aiguard": aiguard_result,
        #        },
        #
        
        action = (aiguard_result.get("action") or "").upper()
        if action != "ALLOW":
            topic_details = (
               aiguard_result.get("detectorResponses", {})
                             .get("topic", {})
                             .get("details", {})
            )

            triggered_topics = topic_details.get("triggered_topics") or topic_details.get("detectedLabelNames") or []
            if isinstance(triggered_topics, str):
                triggered_topics = [triggered_topics]

            topics_str = ", ".join(triggered_topics) if triggered_topics else ""

            return _response(
                403,
                {
                    "answer": f"Request blocked by AI Guard policy {topics_str}",
                },
            )

        # 2) If allowed, proceed to Bedrock retrieve_and_generate
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

        return _response(
            200,
            {
                "answer": answer,
                "sessionId": new_session_id,
                "citations": citations,
                "aiguard": aiguard_result,
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