import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from vector_embedding_db import save_vector, persist

load_dotenv()


def text_to_embedding(text: str):
    """Convert text into an embedding vector."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        output_dimensionality=128,
    )
    return embeddings.embed_query(text)


def save_resume_embeddings(resume_data: dict):
    """Save each resume section as one embedding document in ChromaDB."""
    email = resume_data.get("email")
    if not email:
        raise ValueError("Resume data must include an email field.")

    saved_ids = []
    for key, value in resume_data.items():
        if key == "email" or value is None:
            continue

        if key == "skills" and isinstance(value, list):
            chunk = " ".join(str(item) for item in value) + " skills"
        elif key == "experience" and isinstance(value, list):
            experience_lines = []
            for item in value:
                if isinstance(item, dict):
                    company = item.get("company", "")
                    role = item.get("role", "")
                    years = item.get("years")
                    parts = []
                    if role:
                        parts.append(role)
                    if company:
                        parts.append(f"at {company}")
                    if years is not None:
                        parts.append(f"for {years} years")
                    experience_lines.append(" ".join(parts).strip())
                else:
                    experience_lines.append(str(item))
            chunk = " ; ".join(experience_lines) + " experience"
        elif key == "projects" and isinstance(value, list):
            project_lines = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    tech = item.get("tech")
                    tech_text = " ".join(tech) if isinstance(tech, (list, tuple)) else str(tech) if tech is not None else ""
                    if name and tech_text:
                        project_lines.append(f"{name} project using {tech_text}")
                    elif name:
                        project_lines.append(f"{name} project")
                    else:
                        project_lines.append(json.dumps(item))
                else:
                    project_lines.append(str(item))
            chunk = " ; ".join(project_lines)
        elif isinstance(value, (list, dict)):
            chunk = json.dumps(value)
        else:
            chunk = str(value)

        if not chunk:
            continue

        document_id = f"{email}_{key}"
        metadata = {"email": email, "section": key}
        vector = text_to_embedding(chunk)

        save_vector(
            vector,
            metadata=metadata,
            document=chunk,
            id=document_id,
        )
        saved_ids.append(document_id)

    persist()
    return saved_ids


if __name__ == "__main__":
    sample_resume = {
        "email": "abc@gmail.com",
        "skills": ["Python", "Django", "SQL"],
        "experience": [{"company": "XYZ", "role": "Backend Developer", "years": 2}],
        "projects": [{"name": "RentRoom", "tech": ["Django", "Postgres"]}],
    }

    saved = save_resume_embeddings(sample_resume)
    print("Saved ids:", saved)
