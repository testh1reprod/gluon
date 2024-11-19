import streamlit as st


def nav_bar():
    """
    Show the top navigation bar
    """
    st.markdown(
        """
    <nav class="navbar navbar-expand-sm navbar-light bg-white fixed-top" style="padding-left: 81px;">
        <div class="navbar-nav">
          <a class="nav-item nav-link active" href="#get-started" style="color: #18A0FB;">Get Started <span class="sr-only">(current)</span></a>
          <a class="nav-item nav-link" href="#quick-demo" style="color: #18A0FB;">Demo </a>
          <a class="nav-item nav-link" href="#features-of-autogluon-assistant" style="color: #18A0FB;">Features </a>
          <a class="nav-item nav-link" href="#run-autogluon" style="color: #18A0FB;">Run</a>
          <a class="nav-item nav-link" href="#preview-dataset" style="color: #18A0FB;">Dataset</a>
          <a class="nav-item nav-link disabled" href="https://github.com/autogluon/autogluon-assistant" style="color: #18A0FB;">Github</a>
        </div>
    </nav>
    """,
        unsafe_allow_html=True,
    )


def main():
    nav_bar()


if __name__ == "__main__":
    main()
