docker compose exec api python3 -c "
import fitz
doc = fitz.open('/code/docs/AWS_DMS_Documentation.pdf')
sample = fitz.open()
sample.insert_pdf(doc, from_page=3, to_page=19)
sample.save('/code/docs/AWS_DMS_sample_pages.pdf')
print('Saved 20-page sample')
"

