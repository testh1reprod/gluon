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
                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit,
                sed quia consequuntur magni dolores eos qui ratione voluptatem sequi
                nesciunt. Neque porro quisquam est.
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
                Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit,
                sed quia consequuntur magni dolores eos qui ratione voluptatem sequi
                nesciunt. Neque porro quisquam est.
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
                Consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.
                Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet,
                consectetur, adipisci velit, sed quia non numquam.
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
                Eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui
                dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non
                numquam eius modi tempora.
            </div>
            <a href="#" class="learn-more-btn">Learn more</a>
        </div>
        """, unsafe_allow_html=True)

def main():
    features()
    st.markdown("---",unsafe_allow_html=True)

if __name__ == "__main__":
    main()