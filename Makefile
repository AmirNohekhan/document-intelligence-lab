.PHONY: test evaluate ask

test:
	pytest

evaluate:
	python -m document_intelligence_lab evaluate --out reports

ask:
	python -m document_intelligence_lab ask "Does claim CLM-1001 appear covered?"

