---
name: write-document
description: Write a structured document (report, letter, proposal, plan) and save it as Word or PDF. Use when asked to write or draft a document to keep or send.
---

# Write a document

1. Make sure you know the purpose, the reader and roughly how long it should be. If one of these is unclear and matters, ask one short question; otherwise make a sensible choice and say what you assumed.
2. Write the document. For anything longer than a letter, plan it as a short outline of headings first.
3. Put the content in the file using only these line formats, which the document tools understand:
   - `# Heading` for main sections
   - `## Subheading` for subsections
   - `- item` for bullet points
   - plain lines for paragraphs (one paragraph per line, no line breaks inside a paragraph)
   Don't use bold, tables, numbered lists or links in the file content; write them as plain text.
4. Save it in the same reply; don't stop to ask whether to save. Use `create_docx` for Word (editable) or `create_pdf` for PDF (to send or print): the format the user asked for, or Word if they didn't say. Give it a clear title, and save to `~/Documents/<short-name>.docx` (or `.pdf`) unless the user named a place. The app asks the user to approve saving, and shows them the content first.
5. Tell the user where it was saved, mention any placeholders they need to fill in (like [Your Name]), and offer changes.

Write plainly: short sentences, the main point first in each section, no filler.
