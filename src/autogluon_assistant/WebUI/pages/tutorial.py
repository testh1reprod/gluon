import streamlit as st
import base64


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = (
        """
    <style>
    .left-section {
        width: 47vw;
        background-image: url("data:image/png;base64,%s");
        background-size: 45vw;
        background-repeat: no-repeat;
        background-position: left top;
        display: flex;
        background-color: #ececec;
        flex-direction: column;
    }
    </style>
    """
        % bin_str
    )
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    set_png_as_page_bg("./static/background.png")
    st.markdown(
        """
    <style>
    body {
          font-family: 'Inter', sans-serif;
        }
    .main-container {
        display: flex;
        height: 70vh;
        flex-direction: row;
        margin-bottom: 10px;
    }
    .right-section {
        flex: 1;
        color: black;
        background-color: #ececec;
        display:flex;
        flex-direction:column;
        justify-content: center;
        align-items: flex-start;
        padding-left: 3vw;
    }
    .logo {
        width: 13rem;
        height: 13rem;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        top: 2rem;
        right: 15%;
        margin-left:2rem;
    }
    .title {
        font-size: 4rem;
        font-weight: bold;
        color: white;
        width: 30vw;
        height: 30vh;
        margin: 4vw;
        margin-top:13vh;
        margin-bottom: 10px;
        font-family: "Inter", sans-serif;
        position: relative;

        
    }
    .titleWithLogo{
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        position: relative;

    }
    .subtitle {
        font-size: 2rem;
        text-align: center;
        max-width: 80%;
        color: white;
        font-family: "Inter", sans-serif;
        position: relative;
        

        
    }
    .get-started-title {
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
        font-family: "Inter", sans-serif;
    }
    .description {
        font-size: 1rem;
        line-height: 24px;
        margin-bottom: 20px;
        font-family: "Inter", sans-serif;
        padding-right: 50px;
    }
    .steps {
        font-size: 20px;
        line-height: 32px;
        font-family: "Inter", sans-serif;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div class="main-container" id="get-started">
        <div class="left-section">
            <div class="titleWithLogo">
                <div class="title">AutoGluon<br>Assistant</div>
                    <div class="logo">
                    <img src="https://auto.gluon.ai/stable/_images/autogluon-s.png" alt="AutoGluon Logo">
                    </div>
                </div>
            <div class="subtitle">Fast and Accurate ML in 0 Lines of Code</div>
        </div>
        <div class="right-section">
            <div class="get-started-title">Get Started</div>
            <div class="description">AutoGluon Assistant (AG-A) provides users a simple interface where they can upload their data, describe their problem, and receive a highly accurate and competitive ML solution — without writing any code. By leveraging the state-of-the-art AutoML capabilities of AutoGluon and integrating them with a Large Language Model (LLM), AG-A automates the entire data science pipeline. AG-A takes AutoGluon’s automation from three lines of code to zero, enabling users to solve new supervised learning tabular problems using only natural language descriptions.</div>
            <div class="steps">
                <ol>
                    <li>Upload dataset files (CSV,XLSX)</li>
                    <li>Provide dataset description</li>
                    <li>Launch AutoGluon Assistant</li>
                </ol>
            </div>    
        </div> 
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
