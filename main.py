# Import
import streamlit as st
import requests
import style
import os

# Title
st.markdown("<h1 style='text-align: center; color: white;'>Cartoonise Me!</h1>", unsafe_allow_html=True)

# Make two columns for content and style to be in the same row
col1, col2 = st.columns(2)

content_choice = st.sidebar.selectbox('Content Image', ('Choose from Library', 'Upload Image'))

if content_choice == "Choose from Library":
    content_name = st.sidebar.selectbox('Select Content Image', ('Octopus', 'RobertPattinson', 'MargotRobbie', 'Fuji',
                                                                 'BradPitt', 'Cat'))
    input_content = "images/content_images/" + content_name + ".jpg"

    col1.markdown("<h3 style='text-align: center; color: white;'>Content Image</h3>", unsafe_allow_html=True)
    content_image = style.load_image(input_content)
    col1.image(style.image_convert(content_image), width=400, use_column_width=True)

elif content_choice == "Upload Image":
    col1.markdown("<h3 style='text-align: center; color: white;'>Content Image</h3>", unsafe_allow_html=True)
    st.subheader("Upload Content Image")
    input_content = st.file_uploader("Upload Images", type=["jpg", "jpeg"])
    if input_content is not None:
        # To See details
        file_details = {"filename": input_content.name, "filetype": input_content.type,
                        "filesize": input_content.size}
        st.write(file_details)

        with open(os.path.join("images/content_images_upload/", input_content.name), "wb") as f:
            f.write(input_content.getbuffer())
        st.success("File Saved!")

        input_content = "images/content_images_upload/" + input_content.name

        content_image = style.load_image(input_content)
        col1.image(style.image_convert(content_image), width=400, use_column_width=True)

style_choice = st.sidebar.selectbox("Style Image", ('Choose from Library', 'Upload Image'))

if style_choice == "Choose from Library":
    style_name = st.sidebar.selectbox('Select Style Image', ('Azuki', 'StarryNight', 'Mosaic', 'Picasso', 'Cubism'))
    input_style = "images/style_images/" + style_name + ".jpg"

    col2.markdown("<h3 style='text-align: center; color: white;'>Style Image</h3>", unsafe_allow_html=True)
    style_image = style.load_image(input_style)
    col2.image(style.image_convert(style_image), width=400, use_column_width=True)

elif style_choice == "Upload Image":
    col2.markdown("<h3 style='text-align: center; color: white;'>Style Image</h3>", unsafe_allow_html=True)
    st.subheader("Upload Style Image")
    input_style = st.file_uploader("Upload Image", type=["jpg", "jpeg"])

    if input_style is not None:
        # To See details
        file_details = {"filename": input_style.name, "filetype": input_style.type,
                        "filesize": input_style.size}
        st.write(file_details)

        style_image = style.load_image(input_style)
        col2.image(style.image_convert(style_image), width=400, use_column_width=True)

checkbox = st.sidebar.checkbox("Toon Me")  # Toggle/Un-toggle Warping

total_iterations = st.sidebar.slider("Total Iterations", 0, 2500, step=100,  # Parameter Slider for total iterations
                                     value=500)
alpha = st.sidebar.text_input("Alpha (Content Weight)", value=0.001)  # Parameter Slider for total iterations
beta = st.sidebar.text_input("Beta (Style Weight)", value=1)  # Parameter Slider for total iterations
learning_rate = st.sidebar.text_input("Learning Rate", value=0.075)  # Parameter Slider for learning rate

col1, col2, col3, col4, col5 = st.columns(5)  # Simulate 5 columns in order for the button to be centered
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3:
    clicked = st.button("Cartoonize Me!")
if clicked:  # If button is clicked, stylize the content image, produce an output
    col1, col2, col3 = st.columns([1, 3, 1])
    if input_content is not None:
        if input_style is not None:
            if checkbox:
                print(input_content)
                col2.markdown("<h3 style='text-align: center; color: white;'>Output Image</h3>", unsafe_allow_html=True)
                url = "https://toonify.p.rapidapi.com/v0/toonify"
                query = {
                    "face_index": 1,
                    "return_aligned": "false",
                }
                headers = {
                    "x-rapidapi-host": "toonify.p.rapidapi.com",
                    "x-rapidapi-key": "cf977fdffemsh2a061850253ea2ap17c2aejsnbd65eb771700",
                    "accept": "image/jpeg"
                }
                files = {"image": open(input_content, "rb")}

                response = requests.request("POST", url, files=files, headers=headers, params=query)

                with open("images/warped_images/warped.jpg", "wb") as f:
                    f.write(response.content)

                input_warped = "images/warped_images/warped.jpg"
                col2.image(style.stylize(input_warped, input_style, total_iterations, float(learning_rate),
                                         float(alpha), float(beta)),
                           width=400, use_column_width=True)
            else:
                col2.image(style.stylize(input_content, input_style, total_iterations, float(learning_rate),
                                         float(alpha), float(beta)),
                           width=400, use_column_width=True)
        else:
            col2.error("Style image has not been uploaded.")
    else:
        col2.error("Content image has not been uploaded.")
