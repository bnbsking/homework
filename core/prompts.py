class Prompts:
    faiss_medical_events = """
        Find the Timeline of medical events: Date | Provider | Reason.
    """

    llm_summary_medical_events = """
        **Goal**: You are a specialist of medical events.
        Given chunks extracted from RAG, if there are any medical events,
        extract each event in "Date, Provider and Reason" in the following JSON format.

        **Notes**: the date should be in "MM-DD-YYYY" format.

        **Chunks**:
        {{ chunks }}

        **Response Format**:
        ```json
        [
            {
                "date": [str],
                "provider": [str],
                "reason": [str],
            },
            ...
        ]
        ```
    """
    
    extract_name = """
    **Goal**: I used regex to filter the keywords with their frequencies.
    As the following: {{ candidates }}.
    Which is mostly likely a person's name?

    **Response Format**:
        ```json
        {
            "name": [str],
            "explanation": [str],
        }
        ```
    """

    extract_aod = """
    **Goal**: I used regex to filter the keywords with their frequencies.
    As the following: {{ candidates }}.
    Which is mostly likely a aod (Admission Date)?
    Output the date in "MM-DD-YYYY" format.

    **Response Format**:
        ```json
        {
            "aod": [str],
            "explanation": [str],
        }
        ```
    """

    extract_dob = """
    **Goal**: I used regex to filter the keywords with their frequencies.
    As the following: {{ candidates }}.
    Which is mostly likely a dob (Date of Birth)?
    Output the date in "MM-DD-YYYY" format.

    **Response Format**:
        ```json
        {
            "dob": [str],
            "explanation": [str],
        }
        ```
    """


def prompt_getter(name: str) -> str:
    return getattr(Prompts, name)
