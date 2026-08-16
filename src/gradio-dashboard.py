import os
import numpy as np
import pandas as pd
import gradio as gr

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# Configuration
# ============================================================

load_dotenv()

PROJECT_ROOT = "/home/charis/Desktop/Projects/Drug-Recommendation-System"

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "Drug_Data.csv"
)

CHROMA_PATH = os.path.join(
    PROJECT_ROOT,
    "src",
    "chroma_db"
)

# IMPORTANT:
# This MUST be the same embedding model that was used
# when the Chroma database was originally created.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("CSV columns:")
print(df.columns.tolist())


# ============================================================
# Check required columns
# ============================================================

required_columns = [
    "drugName",
    "Prescribed_for",
    "User_Rating",
    "Count_of_Reviews"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The CSV is missing the following columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# Clean dataset
# ============================================================

df["drugName"] = (
    df["drugName"]
    .fillna("")
    .astype(str)
)

df["Prescribed_for"] = (
    df["Prescribed_for"]
    .fillna("")
    .astype(str)
)

df["User_Rating"] = pd.to_numeric(
    df["User_Rating"],
    errors="coerce"
)

df["Count_of_Reviews"] = pd.to_numeric(
    df["Count_of_Reviews"],
    errors="coerce"
)


# ============================================================
# Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
)


# ============================================================
# Load ChromaDB
# ============================================================

print("Loading ChromaDB...")

db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

print("ChromaDB loaded successfully.")


# ============================================================
# Retrieve recommendations
# ============================================================
def retrieve_drug_recommendations(
    query,
    condition="All",
    initial_top_k=200,
    final_top_k=10
):

    if not query or not query.strip():
        return pd.DataFrame()

    # ========================================================
    # Build query
    # ========================================================

    search_query = query.strip()

    if condition != "All":
        search_query = (
            f"{condition}. {search_query}"
        )

    # ========================================================
    # Search Chroma
    # ========================================================

    results = db.similarity_search_with_score(
        search_query,
        k=initial_top_k
    )

    rows = []

    # ========================================================
    # Extract information from page_content
    # ========================================================

    import re

    for doc, score in results:

        content = doc.page_content.strip()

        # Example content:
        #
        # : 30983
        # 0: Viibryd Depression "I had underlying anxiety..." 10

        lines = content.splitlines()

        if not lines:
            continue

        # Find the actual data line
        data_line = None

        for line in lines:

            if re.match(r"^\d+:\s*", line):
                data_line = line
                break

        if data_line is None:
            continue

        # Remove row prefix
        data_line = re.sub(
            r"^\d+:\s*",
            "",
            data_line
        ).strip()

        # ====================================================
        # Extract rating
        # ====================================================

        rating_match = re.search(
            r"\s+(\d+(?:\.\d+)?)\s*$",
            data_line
        )

        if rating_match:

            rating = float(
                rating_match.group(1)
            )

            text_without_rating = (
                data_line[
                    :rating_match.start()
                ].strip()
            )

        else:

            rating = None
            text_without_rating = data_line

        # ====================================================
        # Extract review
        # ====================================================

        review_match = re.search(
            r'"(.*)"',
            text_without_rating
        )

        if review_match:

            review = review_match.group(1)

            drug_condition = (
                text_without_rating[
                    :review_match.start()
                ].strip()
            )

        else:

            review = ""

            drug_condition = (
                text_without_rating
            )

        # ====================================================
        # Separate drug name and condition
        #
        # We use the dataset to identify the condition.
        # ====================================================

        drug_name = ""
        prescribed_for = ""

        # ----------------------------------------------------
        # First try selected condition
        # ----------------------------------------------------

        if condition != "All":

            condition_position = (
                drug_condition.lower().find(
                    condition.lower()
                )
            )

            if condition_position >= 0:

                drug_name = (
                    drug_condition[
                        :condition_position
                    ].strip()
                )

                prescribed_for = (
                    drug_condition[
                        condition_position:
                    ].strip()
                )

        # ----------------------------------------------------
        # If condition is All, use known conditions
        # ----------------------------------------------------

        if not drug_name:

            sorted_conditions = sorted(
                [
                    c
                    for c in conditions
                    if c != "All"
                ],
                key=len,
                reverse=True
            )

            lower_text = drug_condition.lower()

            for known_condition in sorted_conditions:

                position = lower_text.find(
                    known_condition.lower()
                )

                if position > 0:

                    drug_name = (
                        drug_condition[
                            :position
                        ].strip()
                    )

                    prescribed_for = (
                        drug_condition[
                            position:
                        ].strip()
                    )

                    break

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if not drug_name:

            parts = drug_condition.split(
                maxsplit=1
            )

            if len(parts) == 2:

                drug_name = parts[0]
                prescribed_for = parts[1]

            else:

                drug_name = drug_condition
                prescribed_for = ""

        # ====================================================
        # Add result
        # ====================================================

        rows.append(
            {
                "drugName": drug_name,
                "Prescribed_for": prescribed_for,
                "User_Rating": rating,
                "Count_of_Reviews": None,
                "distance": float(score),
                "review": review
            }
        )

    # ========================================================
    # Create DataFrame
    # ========================================================

    results_df = pd.DataFrame(rows)

    if results_df.empty:
        return results_df

    # ========================================================
    # Clean
    # ========================================================

    results_df["drugName"] = (
        results_df["drugName"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    results_df["Prescribed_for"] = (
        results_df["Prescribed_for"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # Remove empty drugs
    # ========================================================

    results_df = results_df[
        results_df["drugName"] != ""
    ]

    if results_df.empty:
        return results_df

    # ========================================================
    # Condition filtering
    # ========================================================

    if condition != "All":

        results_df = results_df[
            results_df["Prescribed_for"]
            .str.contains(
                condition,
                case=False,
                na=False,
                regex=False
            )
        ]

    if results_df.empty:
        return results_df

    # ========================================================
    # Numeric values
    # ========================================================

    results_df["User_Rating"] = pd.to_numeric(
        results_df["User_Rating"],
        errors="coerce"
    )

    # ========================================================
    # Group by drug
    # ========================================================

    drug_results = (
        results_df
        .groupby("drugName", as_index=False)
        .agg(
            distance=("distance", "mean"),
            rating=("User_Rating", "mean"),
            condition=("Prescribed_for", "first")
        )
    )

    if drug_results.empty:
        return drug_results

    # ========================================================
    # Rating
    # ========================================================

    drug_results["rating"] = (
        drug_results["rating"]
        .fillna(0)
    )

    drug_results["rating_score"] = (
        drug_results["rating"]
        .clip(0, 10)
        / 10
    )

    # ========================================================
    # Semantic score
    # ========================================================

    min_distance = (
        drug_results["distance"].min()
    )

    max_distance = (
        drug_results["distance"].max()
    )

    distance_range = (
        max_distance - min_distance
    )

    if distance_range > 0:

        drug_results["semantic_score"] = (
            1
            -
            (
                drug_results["distance"]
                - min_distance
            )
            / distance_range
        )

    else:

        drug_results["semantic_score"] = 1.0

    # ========================================================
    # Final score
    # ========================================================

    drug_results["final_score"] = (
        0.80 * drug_results["semantic_score"]
        +
        0.20 * drug_results["rating_score"]
    )

    # ========================================================
    # Sort
    # ========================================================

    drug_results = drug_results.sort_values(
        by="final_score",
        ascending=False
    )

    return drug_results.head(
        int(final_top_k)
    )

# ============================================================
# Recommendation function
# ============================================================
def recommend_drugs(
    query,
    condition,
    top_k
):

    if not query or not query.strip():

        return pd.DataFrame(
            columns=[
                "Drug",
                "Condition",
                "Rating",
                "Score"
            ]
        )

    recommendations = retrieve_drug_recommendations(
        query=query,
        condition=condition,
        initial_top_k=200,
        final_top_k=int(top_k)
    )

    if recommendations.empty:

        return pd.DataFrame(
            {
                "Message": [
                    "No relevant results were found."
                ]
            }
        )

    output = recommendations.rename(
        columns={
            "drugName": "Drug",
            "condition": "Condition",
            "rating": "Rating",
            "final_score": "Score"
        }
    )

    output["Rating"] = (
        output["Rating"]
        .round(2)
    )

    output["Score"] = (
        output["Score"]
        .round(4)
    )

    return output[
        [
            "Drug",
            "Condition",
            "Rating",
            "Score"
        ]
    ]

# ============================================================
# Conditions
# ============================================================

conditions = sorted(
    df["Prescribed_for"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

conditions = [
    "All"
] + [
    condition
    for condition in conditions
    if condition.strip()
]


# ============================================================
# Gradio Dashboard
# ============================================================

with gr.Blocks() as dashboard:

    gr.Markdown(
        """
        # 💊 Drug Information Retrieval System

        Enter a description of a condition or treatment
        experience to find semantically similar drug reviews.

        **For informational and research purposes only.  
        This system does not provide medical diagnosis
        or treatment recommendations.**
        """
    )


    # ========================================================
    # Search controls
    # ========================================================

    with gr.Row():

        with gr.Column(
            scale=3
        ):

            user_query = gr.Textbox(
                label="Describe your situation",

                placeholder=(
                    "e.g. I am interested in people's "
                    "experiences with depression treatments"
                ),

                lines=4
            )


        with gr.Column(
            scale=1
        ):

            condition_dropdown = gr.Dropdown(
                choices=conditions,

                value="All",

                label="Condition"
            )


            top_k = gr.Slider(
                minimum=3,

                maximum=20,

                value=10,

                step=1,

                label="Number of results"
            )


    # ========================================================
    # Search button
    # ========================================================

    search_button = gr.Button(
        "🔎 Find relevant drugs",

        variant="primary"
    )


    # ========================================================
    # Results
    # ========================================================

    gr.Markdown(
        "## Results"
    )


    output = gr.Dataframe(
        headers=[
            "Drug",
            "Condition",
            "Rating",
            "Reviews",
            "Score"
        ],

        datatype=[
            "str",
            "str",
            "number",
            "number",
            "number"
        ],

        interactive=False
    )


    # ========================================================
    # Button event
    # ========================================================

    search_button.click(
        fn=recommend_drugs,

        inputs=[
            user_query,
            condition_dropdown,
            top_k
        ],

        outputs=output
    )


    # ========================================================
    # Enter key event
    # ========================================================

    user_query.submit(
        fn=recommend_drugs,

        inputs=[
            user_query,
            condition_dropdown,
            top_k
        ],

        outputs=output
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    dashboard.launch(
        theme=gr.themes.Soft()
    )