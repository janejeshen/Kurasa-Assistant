# Kurasa-Assistant
# Kurasa Assistant

A small, sharp Rasa assistant that answers “how do I…?” questions for **Kurasa**.  
It reads a local guide, turns it into friendly, step-by-step replies, and can chat over **REST** or **Telegram**.


## Why this exists

- People ask the same “where do I click?” questions.
- You already have the answers in a doc.
- This bot finds the right section and replies like a helpful colleague, not a manual.


## Highlights

- **Humanized steps** – numbered, short, and clear (no arrow soup).
- **Local knowledge base** – `actions/kurasa_guide.txt` (split by `###` headings).
- **Smarts without dependence** – works offline from the guide; can optionally call an external paraphraser.
- **Short-term memory** – keeps the last few Q/A turns so follow-ups make sense.
- **Telegram ready** – simple webhook setup (ngrok supported).
- **Tests included** – quick confidence when you change code.



