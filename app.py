"""
Streamlit Application for OCR Text Extraction
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.preprocessing import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.text_extraction import TextExtractor
from src.utils import save_results, calculate_accuracy


def main():
    st.set_page_config(
        page_title="OCR Shipping Label Extractor",
        page_icon="📦",
        layout="wide"
    )
    
    # Title and description
    st.title("📦 OCR Shipping Label Text Extractor")
    st.markdown("""
    Upload shipping label images to extract text containing the **`_1_`** pattern."""
    
    )
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    show_preprocessing = st.sidebar.checkbox("Show preprocessing steps", value=False)
    # show_all_text = st.sidebar.checkbox("Show full extracted text", value=False)
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold (%)", 
        min_value=0, 
        max_value=100, 
        value=75
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a shipping label image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload a clear image of the shipping label"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            # Save temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
    
    with col2:
        st.header("🔍 Extraction Results")
        
        if uploaded_file is not None:
            # Process button
            if st.button("🚀 Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Initialize components
                        preprocessor = ImagePreprocessor()
                        ocr_engine = OCREngine()
                        extractor = TextExtractor()
                        
                        # Step 1: Preprocess
                        st.info("Step 1: Preprocessing image...")
                        variants = preprocessor.preprocess(tmp_path)
                        
                        if show_preprocessing:
                            st.subheader("Preprocessing Variants")
                            preview_cols = st.columns(len(variants))
                            for idx, variant in enumerate(variants[:(len(variants))]):
                                with preview_cols[idx % (len(variants))]:
                                    st.image(
                                        variant, 
                                        caption=f"Variant {idx+1}",
                                        use_container_width=True,
                                        channels="GRAY"
                                    )
                        
                        # Step 2: OCR
                        st.info("Step 2: Performing OCR...")
                        ocr_text, ocr_confidence = ocr_engine.extract_text_optimized(variants)
                        
                        # if show_all_text:
                        #     with st.expander("📄 Full Extracted Text"):
                        #         st.text(ocr_text)
                        
                        # Step 3: Extract target
                        st.info("Step 3: Extracting target pattern...")
                        all_ocr_results = ocr_engine.extract_text_all_methods(variants)
                        target_text, extraction_confidence = extractor.batch_extract(all_ocr_results)
                        
                        # Display results
                        st.success("✅ Processing complete!")
                        
                        if target_text:
                            # Calculate final confidence
                            final_confidence = (ocr_confidence * 0.4 + extraction_confidence * 0.6)
                            
                            # Display target in success box
                            st.markdown("### 🎯 Target Line Found")
                            st.code(target_text, language=None)
                            
                            # Confidence metrics
                            metric_cols = st.columns(3)
                            with metric_cols[0]:
                                st.metric("OCR Confidence", f"{ocr_confidence:.1f}%")
                            with metric_cols[1]:
                                st.metric("Extraction Confidence", f"{extraction_confidence:.1f}%")
                            with metric_cols[2]:
                                st.metric("Overall Confidence", f"{final_confidence:.1f}%")
                            
                            # Confidence indicator
                            if final_confidence >= confidence_threshold:
                                st.success(f"✅ High confidence extraction (>{confidence_threshold}%)")
                            else:
                                st.warning(f"⚠️ Low confidence extraction (<{confidence_threshold}%)")
                            
                            # Download results
                            results = {
                                'filename': uploaded_file.name,
                                'timestamp': datetime.now().isoformat(),
                                'target_text': target_text,
                                'ocr_confidence': ocr_confidence,
                                'extraction_confidence': extraction_confidence,
                                'final_confidence': final_confidence,
                                'full_text': ocr_text
                            }
                            
                            st.download_button(
                                label="📥 Download Results (JSON)",
                                data=json.dumps(results, indent=2),
                                file_name=f"ocr_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        else:
                            st.error("❌ Target pattern `_1_` not found in image")
                            st.info("💡 Try uploading a clearer image or check if the pattern exists")
                            
                            # Show extracted text for debugging
                            with st.expander("🔍 Debug: Show extracted text"):
                                st.text(ocr_text)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {str(e)}")
                        import traceback
                        with st.expander("🐛 Error Details"):
                            st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload an image to begin extraction")
    
    # # Batch processing section
    # st.markdown("---")
    # st.header("📊 Batch Processing")
    
    # batch_files = st.file_uploader(
    #     "Upload multiple images for batch processing",
    #     type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    #     accept_multiple_files=True
    # )
    
    # if batch_files:
    #     if st.button("🚀 Process Batch", type="primary"):
    #         results_list = []
    #         progress_bar = st.progress(0)
    #         status_text = st.empty()
            
    #         # Initialize components once
    #         preprocessor = ImagePreprocessor()
    #         ocr_engine = OCREngine()
    #         extractor = TextExtractor()
            
    #         for idx, file in enumerate(batch_files):
    #             status_text.text(f"Processing {idx+1}/{len(batch_files)}: {file.name}")
                
    #             try:
    #                 # Save temporary file
    #                 with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
    #                     tmp_file.write(file.getvalue())
    #                     tmp_path = tmp_file.name
                    
    #                 # Process
    #                 variants = preprocessor.preprocess(tmp_path)
    #                 ocr_text, ocr_conf = ocr_engine.extract_text_optimized(variants)
    #                 all_results = ocr_engine.extract_text_all_methods(variants)
    #                 target, ext_conf = extractor.batch_extract(all_results)
                    
    #                 results_list.append({
    #                     'filename': file.name,
    #                     'target_text': target if target else "NOT FOUND",
    #                     'confidence': (ocr_conf * 0.4 + ext_conf * 0.6) if target else 0.0,
    #                     'status': 'SUCCESS' if target else 'FAILED'
    #                 })
                    
    #                 # Clean up
    #                 os.unlink(tmp_path)
                    
    #             except Exception as e:
    #                 results_list.append({
    #                     'filename': file.name,
    #                     'target_text': "ERROR",
    #                     'confidence': 0.0,
    #                     'status': f'ERROR: {str(e)}'
    #                 })
                
    #             progress_bar.progress((idx + 1) / len(batch_files))
            
    #         status_text.text("✅ Batch processing complete!")
            
    #         # Display results table
    #         st.subheader("📋 Batch Results")
    #         st.table(results_list)
            
    #         # Calculate accuracy
    #         successful = sum(1 for r in results_list if r['status'] == 'SUCCESS')
    #         accuracy = (successful / len(batch_files)) * 100
            
    #         st.metric("Batch Accuracy", f"{accuracy:.1f}%")
            
    #         # Download batch results
    #         st.download_button(
    #             label="📥 Download Batch Results",
    #             data=json.dumps(results_list, indent=2),
    #             file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    #             mime="application/json"
    #         )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Built with Streamlit | OCR powered by Tesseract | Image processing with OpenCV</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()