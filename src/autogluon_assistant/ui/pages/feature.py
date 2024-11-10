import streamlit as st


def features():
    st.markdown(
        """
        <h1 style='
            font-weight: light;
            padding-left: 20px;
            padding-right: 20px;
            margin-left:60px;
            font-size: 2em;
        '>
            Features of AutoGluon Assistant
        </h1>
    """,
        unsafe_allow_html=True,
    )
    col1, col2, col3, col4 = st.columns([1, 10, 10, 1])
    # Feature 1
    with col2:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">LLM based Task Understanding</div>
            <div class="feature-description">
                Leverage the power of Large Language Models to automatically interpret and understand data science tasks. 
                Autogluon Assistant analyses user’s task description and dataset files, translating them into actionable machine learning objectives without manual intervention.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature 2
    with col3:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Automated Feature Engineering</div>
            <div class="feature-description">
                Streamline your data preparation process with our advanced automated feature engineering.
                Our AI identifies relevant features, handles transformations, and creates new meaningful variables,
                significantly reducing time spent on data preprocessing.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Feature 3
    with col2:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Powered by AutoGluon Tabular</div>
            <div class="feature-description">
                Benefit from the robust capabilities of AutoGluon Tabular, 
                State of the Art AutoML framework. AutoGluon automatically trains and tunes a diverse set of models for your tabular data,
                ensuring optimal performance without the need for extensive machine learning expertise.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="feature-container">
            <div class="feature-title">Coming Soon</div>
            <div class="feature-description">
                Exciting new features are on the horizon! Our team is working on innovative capabilities 
                to enhance your AutoML experience. Stay tuned for updates that will further simplify 
                and improve your machine learning workflow.
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    features()
    st.markdown("---", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
