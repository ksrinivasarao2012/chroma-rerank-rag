import fitz  # PyMuPDF
import base64
import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from backend.core.config import SETTINGS

logger = logging.getLogger(__name__)

class DocumentProcessor:

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100): 
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        groq_key = SETTINGS.GROQ_API_KEY
        if groq_key:
            self.vision_client = ChatGroq(
                api_key=groq_key,
                model_name="llama-3.2-11b-vision-instruct",
                temperature=0.2,
                max_tokens=300
            )
        else:
            self.vision_client = None
            logger.warning("GROQ_API_KEY missing in SETTINGS. Image vision processing disabled.")
        logger.info(f"Initialized DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")


    def _summarize_image(self, image_bytes: bytes) -> str:
        """Sends extracted image bytes to Groq Vision model to produce a detailed text description."""
        if not self.vision_client:
            return ''
        
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": (
                            "Describe this image in detail. If it is a chart, graph, table, or architecture diagram, "
                            "extract all key data points, labels, numbers, and structural relationships."
                        )
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            )

            response = self.vision_client.invoke([message])
            return f"\n\n[EMBEDDED IMAGE DESCRIPTION]:\n{response.content}\n\n"
        except Exception as e:
            logger.error(f"Failed to process image with Groq Vision model:{e}")
            return ''

    def _extract_text(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Reads a raw PDF byte stream in memory and extracts text page by page.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        num_pages = len(doc)
        logger.debug(f"File {filename} has {num_pages} pages")
        
        extracted_pages = []
        
        # Loop through each page, extract the text, and record the page number
        for page_num in range(num_pages):
            page = doc[page_num]
            # 1. Extract standard page text
            raw_text = page.get_text()
            clean_text = raw_text.replace('\n', ' ').strip() if raw_text else ""

            # 2. Detect & process embedded images
            image_list = page.get_images(full=True)
            image_descriptions = ""

            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image['image']

                # Filter out small noise graphics/icons (skip images < 5 KB)
                if len(image_bytes) > 5120:
                    logger.info(f"Processing image on page {page_num + 1} of {filename}")
                    summary = self._summarize_image(image_bytes)
                    if summary:
                        image_descriptions += summary

                # Combine page text with its image summaries
            combined_text = (clean_text + image_descriptions).strip()
            
            if combined_text:
                extracted_pages.append({
                    "text": combined_text,
                    "metadata": {
                        "source_file": filename,
                        "page_number": page_num + 1
                    }
                })
                logger.debug(f"Successfully extracted {len(clean_text)} characters from page {page_num + 1}")
            else:
                logger.warning(f"No text found on page {page_num + 1} of {filename}")
                
        logger.info(f"Finished text extraction. Extracted {len(extracted_pages)} non-empty pages from {filename}")
        return extracted_pages
    
    def _chunk_text(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Splits extracted page strings (text + image summaries) into specified chunk sizes."""
        logger.info(f"Splitting {len(pages_data)} pages of text into chunks")
        chunker = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        final_chunks = []

        for page_data in pages_data:
            text = page_data["text"]
            metadata = page_data["metadata"]
            split_docs = chunker.split_text(text)

            for chunk in split_docs:
                final_chunks.append(
                    {"text": chunk,
                     "metadata": metadata
                    }
                )
                
        logger.info(f"Created {len(final_chunks)} total text chunks")
        return final_chunks
            
    def process_stream(self, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        The main public method. Orchestrates the extraction and chunking of a PDF byte stream.
        """
        logger.info(f"Beginning processing stream for: {filename}")
        # Step 1: Extract text page by page
        extracted_pages = self._extract_text(file_bytes=file_bytes, filename=filename)
        
        # Step 2: Split the pages into chunks
        chunked_data = self._chunk_text(pages_data=extracted_pages)
        
        logger.info(f"Successfully processed stream for {filename} resulting in {len(chunked_data)} chunks")
        # Return the final ready-to-embed data
        return chunked_data