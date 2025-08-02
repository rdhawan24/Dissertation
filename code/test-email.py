import email
from email.message import EmailMessage

def get_email_body(raw_email: str) -> str:
    """
    Extracts the body from a raw email string.

    This function parses the email and finds the most suitable
    text part for the body, prioritizing plain text over HTML.
    It handles multipart messages and decodes the content.

    Args:
        raw_email: A string containing the full raw email source.

    Returns:
        A string with the extracted email body, or an empty
        string if no suitable body is found.
    """
    # Parse the raw email into a message object
    msg = email.message_from_string(raw_email)
    
    body = ""

    # Check if the email is multipart (contains multiple parts)
    if msg.is_multipart():
        # Walk through all parts of the email
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            # Look for a 'text/plain' part that is not an attachment
            if content_type == "text/plain" and "attachment" not in content_disposition:
                try:
                    # Decode the payload and set it as the body
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                    # Prioritize plain text, so we can stop here
                    return body.strip()
                except (AttributeError, TypeError, UnicodeDecodeError):
                    # Handle cases where payload is not a string or has decoding issues
                    continue

        # If no plain text found, fall back to the first HTML part
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if content_type == "text/html" and "attachment" not in content_disposition:
                try:
                    charset = part.get_content_charset() or 'utf-8'
                    body = part.get_payload(decode=True).decode(charset, errors='replace')
                    # Take the first HTML part found
                    return body.strip()
                except (AttributeError, TypeError, UnicodeDecodeError):
                    continue

    # If not multipart, the payload is the body
    else:
        try:
            charset = msg.get_content_charset() or 'utf-8'
            body = msg.get_payload(decode=True).decode(charset, errors='replace')
        except (AttributeError, TypeError, UnicodeDecodeError):
            body = "" # Unable to decode or get payload

    return body.strip()

# --- Example Usage ---

# A sample raw email with both plain text and HTML parts
raw_email_example = """From: sender@example.com
To: recipient@example.com
Subject: Meeting Tomorrow
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary-string"

--boundary-string
Content-Type: text/plain; charset="utf-8"
Content-Transfer-Encoding: 7bit

Hi Team,

This is a reminder about our meeting tomorrow at 10 AM.

See you there,
Sender

--
foobar

--boundary-string
Content-Type: text/html; charset="utf-8"
Content-Transfer-Encoding: 7bit

<html>
  <body>
    <p>Hi Team,</p>
    <p>This is a reminder about our <b>meeting tomorrow at 10 AM</b>.</p>
    <p>See you there,<br>Sender</p>
  </body>
</html>

--boundary-string--
"""

# Extract and print the body
email_body = get_email_body(raw_email_example)
print(email_body)
