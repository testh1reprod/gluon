import streamlit as st
from streamlit_navigation_bar import st_navbar
from autogluon_assistant.Streamlit.style.style import styles,options
from streamlit_card import card
from streamlit_extras.add_vertical_space import add_vertical_space


st.set_page_config(page_title="Home",page_icon="https://pbs.twimg.com/profile_images/1373809646046040067/wTG6A_Ct_400x400.png", layout="wide",initial_sidebar_state="collapsed")
page = st_navbar(["Home","Run Autogluon","Dataset"], selected="Home",styles=styles,options=options)

if page == "Dataset":
    st.switch_page("pages/preview.py")
if page == "Run Autogluon":
    st.switch_page("pages/task.py")
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)

with open('home_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# Function to display the tooltip on hover for <i> element
def display_tooltip_on_icon(icon_class, tooltip_text, i):
    # Generate unique class names for each icon
    hover_class = f'hoverable_icon_{i}'
    tooltip_class = f'tooltip_icon_{i}'

    # Define the unique CSS for each icon
    hover_css = f'''
        .{hover_class} {{
            position: relative;
            display: inline-block;
            cursor: pointer;
            font-size: 50px;
            margin: 20px;
            color: #023e8a;
        }}
        .{hover_class} .{tooltip_class} {{
            opacity: 0;
            position: absolute;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            transition: opacity 0.5s;
            background-color: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 4px;
            border-radius: 4px;
            text-align: center;
            white-space: nowrap;
            z-index: 999;
        }}
        .{hover_class}:hover .{tooltip_class} {{
            opacity: 1;
        }}
    '''
    tooltip_css = f"<style>{hover_css}</style>"

    icon_hover = f'''
        <div class="{hover_class}">
            <i class="{icon_class}"></i>
            <div class="{tooltip_class}">{tooltip_text}</div>
        </div>
    '''
    st.markdown(f'<p>{icon_hover}{tooltip_css}</p>', unsafe_allow_html=True)


def tutorial():
    with st.container(border=True):
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-gear" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col2:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-cloud-arrow-up" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col3:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-diagram-project" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col4:
            st.html(
                """
                 <div style="display: flex; align-items: center; justify-content: center; height: 100%;">
                <i class="fa-solid fa-lightbulb" style="font-size: 50px; margin-top: 30px;color: #023e8a;"></i>
                </div>
                """
            )
        with col5:
            if st.button("Run your first Autogluon Task"):
                st.switch_page("pages/task.py")

def video():
    co1,col2,col3 = st.columns([1,4,1])
    with col2:
        st.video("https://www.youtube.com/watch?v=KNAWp2S3w94")
def demo():
        col1,col2=st.columns(2)
        with col1:
            video()
        with col2:
            card(
                title="Learn more about Autogluon",
                text="Fast and Accurate ML in 3 Lines of Code",
                image="https://miro.medium.com/v2/resize:fit:1400/0*nk1kQ3nBqgATwUOb.png",
                url="https://auto.gluon.ai/stable/index.html",
                styles={
                    "card": {
                        "padding":"0",
                        "width": "70%",
                        "height": "450px",
                        "border-radius": "60px",
                        "box-shadow": "0 0 10px rgba(0,0,0,0.5)",
                    }})

def main():

    demo()
    st.markdown("---")
    tutorial()
    # icon_list = [
    #     {"icon_class": "fa-solid fa-gear", "tooltip": "Lightbulb Icon Tooltip"},
    #     {"icon_class": "fa-solid fa-cloud-arrow-up", "tooltip": "Star Icon Tooltip"}
    # ]
    #
    # for i, icon_info in enumerate(icon_list):
    #     display_tooltip_on_icon(icon_info["icon_class"], icon_info["tooltip"], i)


if __name__ == "__main__":
    main()