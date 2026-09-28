import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page settings
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

# Load trained YOLO model
model = YOLO("best.pt")

# Title
st.title("Helmet and No-Helmet Rider Detection")
st.write("YOLOv8-based Computer Vision Object Detection System")

st.divider()

# Sidebar
st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("""
### Model Information

**Model:** YOLOv8n

**Classes:**
- With Helmet
- Without Helmet
- licence
""")

# Image upload
uploaded_file = st.file_uploader(
    "Upload a road/rider image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # Original image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Detection button
    if st.button("Detect Objects", type="primary"):

        # Run YOLO
        results = model.predict(
            source=np.array(image),
            imgsz=640,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        # Generate detection image
        annotated_image = result.plot()

        # Display result
        with col2:
            st.subheader("Detection Result")
            st.image(
                annotated_image,
                channels="BGR",
                use_container_width=True
            )

        st.divider()

        # Detection statistics
        st.subheader("Detection Summary")

        helmet_count = 0
        no_helmet_count = 0
        licence_count = 0

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name == "With Helmet":
                helmet_count += 1

            elif class_name == "Without Helmet":
                no_helmet_count += 1

            elif class_name == "licence":
                licence_count += 1

        c1, c2, c3 = st.columns(3)

        c1.metric("With Helmet", helmet_count)
        c2.metric("Without Helmet", no_helmet_count)
        c3.metric("Licence", licence_count)

        # Detection details
        if len(result.boxes) == 0:

            st.warning("No objects detected.")

        else:

            st.subheader("Detection Details")

            detections = []

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence_score = float(box.conf[0])

                detections.append({
                    "Detected Class": model.names[class_id],
                    "Confidence": f"{confidence_score:.2%}"
                })

            st.dataframe(
                detections,
                use_container_width=True,
                hide_index=True
            )

else:

    st.info(
        "Upload an image above to start helmet detection."
    )