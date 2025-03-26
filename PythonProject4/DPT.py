import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    st.title("Data Preview Table")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data Preview")
        st.write(df.head())

        selected_columns = st.multiselect("Select columns to include", df.columns.tolist(), default=df.columns.tolist())
        df = df[selected_columns]


        sort_column = st.selectbox("Select column to sort by", df.columns.tolist())
        sort_ascending = st.radio("Sort order", ("Ascending", "Descending"))
        df = df.sort_values(by=sort_column, ascending=(sort_ascending == "Ascending"))


        if st.checkbox("Apply Standard Scaling"):
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['float64', 'int64'])),
                                     columns=df.select_dtypes(include=['float64', 'int64']).columns)
            df.update(df_scaled)


        if st.checkbox("Apply PCA"):
            n_components = st.slider("Select number of components", 1, min(df.shape[1], 5), 2)
            pca = PCA(n_components=n_components)
            df_pca = pd.DataFrame(pca.fit_transform(df.select_dtypes(include=['float64', 'int64'])),
                                  columns=[f'PC{i + 1}' for i in range(n_components)])
            st.write("### PCA Transformed Data")
            st.write(df_pca.head())

        st.write("### Processed Data Preview")
        st.write(df.head())


if __name__ == "__main__":
    main()

#github pull request


