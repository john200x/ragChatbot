import json
import os
import boto3
from botocore.exceptions import ClientError

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime")

KNOWLEDGE_BASE_ID = os.environ["KNOWLEDGE_BASE_ID"]
MODEL_ARN = os.environ["MODEL_ARN"]


def build_system_prompt():
    return (
        "You are a helpful assistant that answers employee questions "
        "using the provided context from the knowledge base. "
        "If the answer is not in the documents, say you don't know."
    )

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
            "body": ""
        }

    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body or "{}")

        query = body.get("query") or body.get("question")
        session_id = body.get("sessionId")

        if not query:
            return _response(
                400,
                {"error": "query (or question) field is required"}
            )

        user_prompt = f"{build_system_prompt()}\n\nQuestion:\n{query}\n\nAnswer:"

        request_params = {
            "input": {"text": user_prompt},
            "retrieveAndGenerateConfiguration": {
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN,
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {
                            "numberOfResults": 5
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