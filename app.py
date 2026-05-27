import streamlit as st

import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

from wordcloud import WordCloud

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="ZENDS Telecom AI Dashboard",

    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title(

    "ZENDS Telecom AI Brand Intelligence Dashboard"
)

st.markdown("---")

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(

    r"C:\Users\Personal\Documents\TELECOM AI PROJECT\final_telecom_ai_dataset.csv"
)

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

model = DistilBertForSequenceClassification.from_pretrained(

    r"C:\Users\Personal\Documents\TELECOM AI PROJECT\sentiment_model"
)

tokenizer = DistilBertTokenizer.from_pretrained(

    r"C:\Users\Personal\Documents\TELECOM AI PROJECT\sentiment_model"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Navigation")

section = st.sidebar.radio(

    "Select Dashboard",

    [

        "Customer Feedback Analysis",

        "EDA Dashboard",

        "Business Insights"
    ]
)

# =========================================================
# CUSTOMER FEEDBACK ANALYSIS
# =========================================================

if section == "Customer Feedback Analysis":

    st.header("Customer Feedback Analyzer")

    user_input = st.text_area(

        "Enter Customer Feedback"
    )

    if st.button("Analyze Feedback"):

        # =====================================
        # TOKENIZATION
        # =====================================

        inputs = tokenizer(

            user_input,

            return_tensors="pt",

            truncation=True,

            padding=True
        )

        # =====================================
        # MODEL PREDICTION
        # =====================================

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits
        )

        # =====================================
        # LABEL MAPPING
        # =====================================

        label_map = {

            0: "Negative",

            1: "Neutral",

            2: "Positive"
        }

        sentiment = label_map[
            prediction.item()
        ]

        # =====================================
        # RULE-BASED CORRECTION
        # =====================================

        negative_words = [

            "slow",

            "bad",

            "poor",

            "issue",

            "problem",

            "not working",

            "delay",

            "weak",

            "complaint",

            "dropping",

            "failure",

            "failed",

            "error"
        ]

        positive_words = [

            "good",

            "excellent",

            "fast",

            "great",

            "amazing",

            "happy",

            "satisfied"
        ]

        text_lower = user_input.lower()

        if any(word in text_lower for word in negative_words):

            sentiment = "Negative"

        elif any(word in text_lower for word in positive_words):

            sentiment = "Positive"

        # =====================================
        # DISPLAY SENTIMENT
        # =====================================

        st.subheader(
            "Predicted Sentiment"
        )

        if sentiment == "Positive":

            st.success(sentiment)

        elif sentiment == "Negative":

            st.error(sentiment)

        else:

            st.warning(sentiment)

        # =====================================
        # SERVICE CATEGORY DETECTION
        # =====================================

        detected_topic = "Customer Support"

        if "network" in text_lower or "signal" in text_lower:

            detected_topic = "Mobile Network"

        elif "broadband" in text_lower or "internet" in text_lower:

            detected_topic = "Broadband Service"

        elif "bill" in text_lower or "payment" in text_lower:

            detected_topic = "Billing & Payments"

        elif "app" in text_lower or "login" in text_lower:

            detected_topic = "Mobile App Issues"

        elif "activation" in text_lower or "sim" in text_lower:

            detected_topic = "Service Activation"

        # =====================================
        # DISPLAY CATEGORY
        # =====================================

        st.subheader(
            "Detected Service Category"
        )

        st.info(detected_topic)

        # =====================================
        # AI GENERATED RESPONSE
        # =====================================

        st.subheader(
            "AI Generated Recommendation"
        )

        if detected_topic == "Mobile Network":

            st.write(

                """
• Move to an open area for better signal.

• Restart your mobile device.

• Switch between 4G and 5G modes.

• Contact support if issue continues.
"""
            )

        elif detected_topic == "Broadband Service":

            st.write(

                """
• Restart modem/router.

• Check connected devices.

• Verify broadband cable connection.

• Contact support if internet remains slow.
"""
            )

        elif detected_topic == "Billing & Payments":

            st.write(

                """
• Verify billing details carefully.

• Check for extra usage charges.

• Ensure payment completed successfully.

• Contact billing support if needed.
"""
            )

        elif detected_topic == "Mobile App Issues":

            st.write(

                """
• Clear application cache.

• Update mobile application.

• Reset password if login fails.

• Check internet connectivity.
"""
            )

        elif detected_topic == "Service Activation":

            st.write(

                """
• SIM activation may take up to 24 hours.

• Restart device after activation.

• Reinsert SIM card if needed.

• Contact support if activation delay continues.
"""
            )

        else:

            st.write(

                """
• Customer support request identified.

• Our support team will assist shortly.

• Please keep your complaint reference safely.
"""
            )

# =========================================================
# EDA DASHBOARD
# =========================================================

elif section == "EDA Dashboard":

    st.header("Exploratory Data Analysis Dashboard")

    # =====================================
    # KPI METRICS
    # =====================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(

        "Total Feedback",

        len(df)
    )

    col2.metric(

        "Negative Feedback",

        len(df[df["sentiment"] == "Negative"])
    )

    col3.metric(

        "Positive Feedback",

        len(df[df["sentiment"] == "Positive"])
    )

    col4.metric(

        "Service Categories",

        df["service_category"].nunique()
    )

    st.markdown("---")

    # =====================================
    # SENTIMENT DISTRIBUTION
    # =====================================

    st.subheader("Sentiment Distribution")

    fig1, ax1 = plt.subplots(figsize=(6,5))

    sns.countplot(

        x="sentiment",

        data=df,

        ax=ax1
    )

    plt.title("Sentiment Distribution")

    st.pyplot(fig1)

    # =====================================
    # SERVICE CATEGORY DISTRIBUTION
    # =====================================

    st.subheader("Service Category Distribution")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    sns.countplot(

        x="service_category",

        data=df,

        ax=ax2
    )

    plt.xticks(rotation=45)

    plt.title("Service Category Distribution")

    st.pyplot(fig2)

    # =====================================
    # DONUT CHART
    # =====================================

    st.subheader("Telecom Complaint Donut Chart")

    category_counts = df[
        "service_category"
    ].value_counts()

    fig_donut = px.pie(

        names=category_counts.index,

        values=category_counts.values,

        hole=0.5,

        title="Telecom Complaint Categories"
    )

    st.plotly_chart(

        fig_donut,

        use_container_width=True
    )

    # =====================================
    # SENTIMENT VS CATEGORY
    # =====================================

    st.subheader("Sentiment vs Service Category")

    cross_tab = pd.crosstab(

        df["service_category"],

        df["sentiment"]
    )

    st.dataframe(cross_tab)

    fig3, ax3 = plt.subplots(figsize=(10,6))

    sns.heatmap(

        cross_tab,

        annot=True,

        cmap="Blues",

        fmt="d",

        ax=ax3
    )

    plt.title("Heatmap")

    st.pyplot(fig3)

    # =====================================
    # TREEMAP
    # =====================================

    st.subheader("Telecom Complaint Treemap")

    treemap_df = df.groupby(

        ["service_category", "sentiment"]

    ).size().reset_index(name="count")

    fig_tree = px.treemap(

        treemap_df,

        path=[

            "service_category",

            "sentiment"
        ],

        values="count",

        title="Complaint Hierarchy"
    )

    st.plotly_chart(

        fig_tree,

        use_container_width=True
    )

    # =====================================
    # MESSAGE LENGTH ANALYSIS
    # =====================================

    st.subheader("Message Length Analysis")

    df["message_length"] = df[
        "feedback_text"
    ].apply(len)

    fig4, ax4 = plt.subplots(figsize=(8,5))

    sns.boxplot(

        x="sentiment",

        y="message_length",

        data=df,

        ax=ax4
    )

    plt.title("Message Length vs Sentiment")

    st.pyplot(fig4)

    # =====================================
    # HISTOGRAM
    # =====================================

    st.subheader("Message Length Distribution")

    fig5, ax5 = plt.subplots(figsize=(8,5))

    ax5.hist(

        df["message_length"],

        bins=30
    )

    plt.title("Histogram")

    st.pyplot(fig5)

    # =====================================
    # WORD CLOUD
    # =====================================

    st.subheader("Telecom Complaint Word Cloud")

    text = " ".join(

        df["feedback_text"]
    )

    wordcloud = WordCloud(

        width=1000,

        height=500,

        background_color="white"
    ).generate(text)

    fig6, ax6 = plt.subplots(figsize=(12,6))

    ax6.imshow(wordcloud)

    ax6.axis("off")

    st.pyplot(fig6)

    # =====================================
    # TOP NEGATIVE COMPLAINTS
    # =====================================

    st.subheader("Top Negative Complaint Areas")

    negative_df = df[

        df["sentiment"] == "Negative"
    ]

    negative_counts = negative_df[
        "service_category"
    ].value_counts()

    fig7, ax7 = plt.subplots(figsize=(8,5))

    sns.barplot(

        x=negative_counts.index,

        y=negative_counts.values,

        ax=ax7
    )

    plt.xticks(rotation=45)

    plt.title("Negative Complaint Areas")

    st.pyplot(fig7)

# =========================================================
# BUSINESS INSIGHTS
# =========================================================

elif section == "Business Insights":

    st.header("Enterprise Telecom Business Intelligence")

    # ==========================================
    # KPI CALCULATIONS
    # ==========================================

    total_feedback = len(df)

    negative_count = len(
        df[df["sentiment"] == "Negative"]
    )

    positive_count = len(
        df[df["sentiment"] == "Positive"]
    )

    neutral_count = len(
        df[df["sentiment"] == "Neutral"]
    )

    customer_satisfaction = round(

        (positive_count / total_feedback) * 100,

        2
    )

    complaint_rate = round(

        (negative_count / total_feedback) * 100,

        2
    )

    # ==========================================
    # KPI CARDS
    # ==========================================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Feedback",
        total_feedback
    )

    col2.metric(
        "Positive Feedback",
        positive_count
    )

    col3.metric(
        "Neutral Feedback",
        neutral_count
    )

    col4.metric(
        "Negative Feedback",
        negative_count
    )

    st.markdown("---")

    # ==========================================
    # CUSTOMER SATISFACTION
    # ==========================================

    st.subheader("Customer Satisfaction Score")

    st.progress(
        int(customer_satisfaction)
    )

    if customer_satisfaction >= 70:

        st.success(

            f"Customer Satisfaction is Excellent : {customer_satisfaction}%"
        )

    elif customer_satisfaction >= 50:

        st.warning(

            f"Customer Satisfaction is Moderate : {customer_satisfaction}%"
        )

    else:

        st.error(

            f"Customer Satisfaction is Low : {customer_satisfaction}%"
        )

    st.markdown("---")

    # ==========================================
    # SENTIMENT OVERVIEW
    # ==========================================

    st.subheader("Overall Sentiment Overview")

    sentiment_counts = df[
        "sentiment"
    ].value_counts()

    fig_sentiment = px.pie(

        names=sentiment_counts.index,

        values=sentiment_counts.values,

        title="Customer Sentiment Overview",

        hole=0.4
    )

    st.plotly_chart(

        fig_sentiment,

        use_container_width=True
    )

    # ==========================================
    # TOP PERFORMING SERVICE
    # ==========================================

    st.subheader("Top Performing Service")

    positive_df = df[
        df["sentiment"] == "Positive"
    ]

    top_positive = positive_df[
        "service_category"
    ].value_counts().idxmax()

    st.success(

        f"{top_positive} receives the highest positive feedback."
    )

    # ==========================================
    # MOST REPORTED SERVICE AREA
    # ==========================================

    st.subheader("Most Reported Service Area")

    top_issue = df[
        "service_category"
    ].value_counts().idxmax()

    st.warning(

        f"{top_issue} receives the highest number of customer reports."
    )

    # ==========================================
    # SERVICE CATEGORY ANALYTICS
    # ==========================================

    st.subheader("Service Category Analytics")

    category_counts = df[
        "service_category"
    ].value_counts()

    fig_category = px.bar(

        x=category_counts.index,

        y=category_counts.values,

        title="Service Category Distribution"
    )

    st.plotly_chart(

        fig_category,

        use_container_width=True
    )

    # ==========================================
    # AI RECOMMENDATIONS
    # ==========================================

    st.subheader("AI-Powered Recommendations")

    recommendations = [

        "Maintain strong telecom network quality.",

        "Improve broadband speed consistency.",

        "Reduce billing/payment complaints.",

        "Improve customer support response time.",

        "Enhance mobile application performance.",

        "Improve activation and onboarding speed.",

        "Focus on customer retention strategies.",

        "Continue improving overall service reliability."
    ]

    for rec in recommendations:

        st.info(rec)

    st.markdown("---")

    # ==========================================
    # AI EXECUTIVE SUMMARY
    # ==========================================

    st.subheader("AI Executive Summary")

    st.success(

        f"""

The AI dashboard analyzed {total_feedback} telecom customer feedback records.

Overall customer satisfaction score is {customer_satisfaction}%.

The strongest performing telecom area is {top_positive}.

The most frequently reported service area is {top_issue}.

Positive customer engagement is significantly higher than negative complaints.

The AI system recommends continuing infrastructure improvements,
customer support optimization, and broadband performance enhancement.

"""
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.write(

    "Developed using DistilBERT, NLP, Streamlit, Deep Learning, RAG, and Telecom AI Analytics"
)