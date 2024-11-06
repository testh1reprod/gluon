import streamlit as st


def features():
    st.markdown("""
        <h1 style='
            font-weight: light;
            padding-left: 20px;
            padding-right: 20px;
            margin-left:60px;
            font-size: 2em;
        '>
            Features of AutoGluon Assistant
        </h1>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .feature-container {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .feature-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .feature-description {
        font-size: 14px;
        color: #666;
        margin-bottom: 15px;
    }
    .learn-more-btn {
        display: inline-block;
        padding: 5px 10px;
        background-color: white;
        color: #18A0FB;
        text-decoration: none;
        border-radius: 3px;
        border: 1px solid #18A0FB;
        font-size: 12px;
        transition: all 0.3s ease;
    }
    .learn-more-btn:hover {
        background-color: #F0F8FF;
    }
    .learn-more-btn:hover, .learn-more-btn:active, .learn-more-btn:focus {
        background-color: #F0F8FF;
        text-decoration: none;
    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2,col3,col4 = st.columns([1,10,10,1])
    # Feature 1
    with col2:
        st.markdown("""
        <div class="feature-container">
            <div class="feature-title">LLM based Task Understanding</div>
            <div class="feature-description">
                Leverage the power of Large Language Models to automatically interpret and understand complex tasks. 
                Autogluon Assistant analyzes user inputs and requirements, translating them into actionable machine learning objectives without manual intervention.
            </div>
            <a href="#" class="learn-more-btn">Learn more</a>
        </div>
        """, unsafe_allow_html=True)

    # Feature 2
    with col3:
        st.markdown("""
        <div class="feature-container">
            <div class="feature-title">Automated Feature Engineering</div>
            <div class="feature-description">
                Streamline your data preparation process with our advanced automated feature engineering.
                Our AI identifies relevant features, handles transformations, and creates new meaningful variables, significantly reducing time spent on data preprocessing.
            </div>
            <a href="#" class="learn-more-btn">Learn more</a>
        </div>
        """, unsafe_allow_html=True)

    # Feature 3
    with col2:
        st.markdown("""
        <div class="feature-container">
            <div class="feature-title">Powered by AutoGluon Tabular</div>
            <div class="feature-description">
                Benefit from the robust capabilities of AutoGluon Tabular, a cutting-edge AutoML framework. 
                Automatically train and tune a diverse set of models for your tabular data, ensuring optimal performance without the need for extensive machine learning expertise.
            </div>
            <a href="#" class="learn-more-btn">Learn more</a>
        </div>
        """, unsafe_allow_html=True)

    # Feature 4
    with col3:
        st.markdown("""
        <div class="feature-container">
            <div class="feature-title">Multi-Table Support</div>
            <div class="feature-description">
               Effortlessly work with complex, relational datasets through our multi-table support feature. 
               Autogluon Assistant intelligently manages relationships between multiple tables, allowing for comprehensive analysis and modeling of intricate data structures.
            </div>
            <a href="#" class="learn-more-btn">Learn more</a>
        </div>
        """, unsafe_allow_html=True)

def main():
    features()
    st.markdown("---",unsafe_allow_html=True)

if __name__ == "__main__":
    main()