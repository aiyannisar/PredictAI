from config import GEMINI_API_KEY

SYSTEM = """You are PredictAI's AI analyst.

Analyze machine-learning predictions clearly and concisely.

The ML model's numerical prediction is the actual model output.
Do not change or invent the prediction.

Explain:
1. What the prediction means
2. Important factors
3. Practical next steps
4. Important limitations

Never claim that a prediction is guaranteed.
"""


def ask_gemini(module, payload, prediction):

    if not GEMINI_API_KEY:
        return {
            "ok": False,
            "text": "Gemini is not configured. Add GEMINI_API_KEY to your .env file."
        }

    try:
        from google import genai

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = f"""
{SYSTEM}

Prediction module:
{module}

User inputs:
{payload}

Machine-learning prediction:
{prediction}

Give a concise, easy-to-understand analysis.
"""

        # IMPORTANT:
        # This is intentionally hard-coded because this model
        # has already been verified to work with your API key.
        interaction = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt
        )

        return {
            "ok": True,
            "text": interaction.output_text.strip()
        }

    except Exception as exc:
        return {
            "ok": False,
            "text": (
                f"Gemini error: {type(exc).__name__}. "
                f"{str(exc)}"
            )
        }