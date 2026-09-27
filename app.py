import streamlit as st
import pandas as pd
import pickle
import json

st.set_page_config(page_title="Netflix Movie Recommender", page_icon="🎬", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    with open('svd_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_data():
    titles = pd.read_csv('movie_titles_clean.csv')
    with open('drop_movie_list.json', 'r') as f:
        drop_list = json.load(f)
    with open('top_customers.json', 'r') as f:
        top_customers = json.load(f)
    with open('all_customer_ids.json', 'r') as f:
        all_ids = set(json.load(f))
    return titles, drop_list, top_customers, all_ids

model = load_model()
titles, drop_list, top_customers, all_ids = load_data()

st.title("🎬 Netflix Movie Recommender")
st.markdown("Get personalized movie recommendations powered by an **SVD collaborative filtering model**, trained on the Netflix Prize dataset.")
st.divider()

st.markdown("### Select a Customer")
tab1, tab2 = st.tabs(["Quick Pick", "Enter ID Manually"])

selected_user = None

with tab1:
    customer_ids = sorted(top_customers)
    quick_pick = st.selectbox("Choose from the 200 most active customers:", customer_ids)
    st.caption("These users have the most ratings, so recommendations for them tend to be most reliable.")
    if st.button("Get Recommendations", type="primary", key="quick_btn"):
        selected_user = quick_pick

with tab2:
    manual_id = st.number_input("Enter any Customer ID from the dataset:", min_value=int(min(all_ids)), max_value=int(max(all_ids)), step=1)
    if st.button("Get Recommendations", type="primary", key="manual_btn"):
        if manual_id in all_ids:
            selected_user = manual_id
        else:
            st.warning("This Customer ID isn't in the dataset. Try another one.")

if selected_user is not None:
    with st.spinner("Finding the best movies for this user..."):
        clean_titles = titles[~titles['Movie_Id'].isin(drop_list)].copy()
        clean_titles['Estimate_Score'] = clean_titles['Movie_Id'].apply(
            lambda x: model.predict(selected_user, x).est
        )
        top_5 = clean_titles.sort_values('Estimate_Score', ascending=False).head(5)

    st.divider()
    st.subheader(f"Top 5 Recommendations for Customer {selected_user}")
    for _, row in top_5.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{row['Name']}** ({int(row['Year'])})")
        with col2:
            st.markdown(f"⭐ {row['Estimate_Score']:.2f}")

st.divider()
st.caption("Built with SVD (matrix factorization) using the `surprise` library. Dataset: Netflix Prize (public, anonymized).")