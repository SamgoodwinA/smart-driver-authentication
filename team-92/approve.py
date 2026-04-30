import imaplib
import email
from email.header import decode_header

# Email configuration
EMAIL_SENDER = "asamgoodwin@gmail.com"
EMAIL_PASSWORD = "khyo uetw qrka vtoe"  # Generated App-Specific Password for Gmail
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Function to check for approval/rejection from replies using IMAP
def check_for_approval():
    try:
        # Connect to the mail server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_SENDER, EMAIL_PASSWORD)
        mail.select("inbox")

        # Search for all messages in the inbox
        status, messages = mail.search(None, '(UNSEEN)')
        if status != "OK":
            print("❌ Error: No new messages.")
            return None

        # Process each unseen email
        for msg_id in messages[0].split():
            status, msg_data = mail.fetch(msg_id, "(RFC822)")
            if status != "OK":
                continue

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or 'utf-8')

                    # Check if the subject contains approval or rejection
                    if "APPROVE" in subject:
                        print("✅ Approval received")
                        return True  # Return True for approval
                    elif "REJECT" in subject:
                        print("❌ Rejection received")
                        return False  # Return False for rejection

        return None  # No response yet

    except Exception as e:
        print(f"❌ Error checking approval: {e}")
        return None


# Example Usage: Periodically check for approval
while True:
    approval_status = check_for_approval()
    if approval_status is not None:
        if approval_status:
            print("Face is approved!")
        else:
            print("Face is rejected!")
    else:
        print("No new approval/rejection responses.")
    
    # Wait before checking again (e.g., 30 seconds)
    time.sleep(30)
