import streamlit as st
import pandas as pd
from pathlib import Path
import os
from OCR import PDFTextExtractor  # Import your PDFTextExtractor class


def init_session_state():
    """Initialize session state variables."""
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def main():
    st.set_page_config(
        page_title="PDF Text Extractor",
        page_icon="📄",
        layout="wide"
    )

    init_session_state()

    st.title("PDF Text Extraction Visualizer")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Extraction Settings")

        # Method selection
        extraction_method = st.radio(
            "Select Text Extraction Method",
            options=["PDFPlumber", "PaddleOCR", "TesseractOCR"],
            help="PDFPlumber works best for searchable PDFs. PaddleOCR and Tesseract work better for scanned documents."
        )

        # Token separation option
        separate_tokens = st.checkbox(
            "Separate Concatenated Tokens",
            value=False,
            help="Fix spacing issues between concatenated words (e.g., 'HelloWorld' → 'Hello World')"
        )

        # Schema selection
        schema_options = ['author', 'title', 'introduction', 'methodology',
                          'related_work', 'experiment', 'background', 'dataset',
                          'acknowledge', 'conclusion', 'result_discussion']
        selected_schema = st.multiselect(
            "Select sections to extract",
            options=schema_options,
            default=schema_options
        )

    # Main content area
    st.header("Upload Files")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])
    with col2:
        json_file = st.file_uploader("Upload JSON layout file", type=['json'])

    # Process button
    if pdf_file and json_file and selected_schema:
        if not st.session_state.processing:
            if st.button("Process PDF"):
                st.session_state.processing = True
                with st.spinner(f'Processing PDF using {extraction_method}...'):
                    try:
                        # Create temp directory if it doesn't exist
                        os.makedirs('temp', exist_ok=True)

                        # Save uploaded files temporarily
                        pdf_path = os.path.join('temp', f"temp_{pdf_file.name}")
                        json_path = os.path.join('temp', f"temp_{json_file.name}")

                        with open(pdf_path, "wb") as f:
                            f.write(pdf_file.getbuffer())
                        with open(json_path, "wb") as f:
                            f.write(json_file.getbuffer())

                        # Process the PDF with selected method
                        ocr_engine = "none"
                        if extraction_method == "PaddleOCR":
                            ocr_engine = "paddle"
                        elif extraction_method == "TesseractOCR":
                            ocr_engine = "tesseract"

                        extractor = PDFTextExtractor(
                            pdf_path,
                            json_path,
                            selected_schema,
                            ocr_engine=ocr_engine,
                            separate_tokens=separate_tokens
                        )
                        layout_data = extractor.process()

                        # Display results
                        st.header("Extracted Text")

                        # Show method used for extraction
                        info_text = f"Text extracted using {extraction_method}"
                        if separate_tokens:
                            info_text += " with token separation"
                        st.info(info_text)

                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs([
                            "Sectional View",
                            "Table View",
                            "Statistics"
                        ])

                        with tab1:
                            for label in selected_schema:
                                sections = [item for item in layout_data if item['label'] == label]
                                if sections:
                                    st.subheader(label.replace('_', ' ').title())
                                    for section in sections:
                                        with st.expander(f"Page {section['page']} - ID {section['id']}"):
                                            st.text_area(
                                                "Extracted Text",
                                                section['text'],
                                                height=150,
                                                key=f"text_{section['id']}"
                                            )
                                            st.info(f"Bounding Box: {section['bbox']}")

                        with tab2:
                            df = pd.DataFrame(layout_data)
                            st.dataframe(df)

                        with tab3:
                            df = pd.DataFrame(layout_data)
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Total Sections", len(df))
                                st.metric("Total Pages", df['page'].nunique())

                            with col2:
                                st.metric(
                                    "Average Text Length",
                                    round(df['text'].str.len().mean(), 2)
                                )
                                st.metric(
                                    "Number of Labels",
                                    df['label'].nunique()
                                )

                        # Download options
                        st.header("Download Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.download_button(
                                label="Download as CSV",
                                data=df.to_csv(index=False),
                                file_name=f"extracted_text_{extraction_method}.csv",
                                mime="text/csv"
                            )

                        with col2:
                            st.download_button(
                                label="Download as JSON",
                                data=df.to_json(orient='records'),
                                file_name=f"extracted_text_{extraction_method}.json",
                                mime="application/json"
                            )

                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")

                    finally:
                        # Clean up temporary files
                        if os.path.exists(pdf_path):
                            os.remove(pdf_path)
                        if os.path.exists(json_path):
                            os.remove(json_path)
                        st.session_state.processing = False


if __name__ == "__main__":
    main()