from langchain_community.document_loaders import PyPDFLoader

def load_pdf_document(file_path:str, custom_metadata: dict = None):
    "point to our document"
    loader = PyPDFLoader(file_path)
    "start loading using load func"
    docs = loader.load()
    
    # Add custom metadata to all documents if provided
    if custom_metadata:
        for doc in docs:
            doc.metadata.update(custom_metadata)
            
    "return our loaded doc"
    return docs

# testing
if __name__ == "__main__":
    docs = load_pdf_document("docs/fake_company.pdf", custom_metadata={"category": "testing"})
    print(docs)
