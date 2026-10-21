# ---- variables you may want to change ----
REMOTE_HOST     := gbar
REMOTE_DIR      := ~/idlcv/project_2/
SCRIPT_LOCATION := ~/idlcv/
OUTPUT_DIR      := /zhome/2b/7/233718/idlcv/output
LOCAL_DIR       := /Users/wojciechwladzinski/projects/uni/introduction-to-dl-in-cv/video_classification_project_2/output

FILE     ?= single_frame_cnn.py
OUT_FILE ?= test.out
ERR_FILE ?= error.err
DIAGRAM1 ?= *_loss.png
DIAGRAM2 ?= *_acc.png
# --------------------------------------------

.PHONY: deploy download logs diagrams

# Upload FILE to the cluster and submit the batch job (same as deploy.sh)
deploy:
	@echo "Uploading $(FILE) to $(REMOTE_HOST):$(REMOTE_DIR)"
	scp "$(FILE)" "$(REMOTE_HOST):$(REMOTE_DIR)"
	@echo "Submitting batch job on $(REMOTE_HOST)"
	REMOTE_CMD="bsub -app c02516_1g.10gb < $(SCRIPT_LOCATION)create_batch_job.sh"; \
	ssh -t "$(REMOTE_HOST)" "bash -lic $$(printf '%q' "$$REMOTE_CMD")"

# Download test.out + error.err from the cluster output directory
logs:
	scp "$(REMOTE_HOST):$(OUTPUT_DIR)/$(OUT_FILE)" "$(LOCAL_DIR)/"
	scp "$(REMOTE_HOST):$(OUTPUT_DIR)/$(ERR_FILE)" "$(LOCAL_DIR)/"

# Download the two diagrams from the cluster output directory
diagrams:
	scp "$(REMOTE_HOST):$(OUTPUT_DIR)/$(DIAGRAM1)" "$(LOCAL_DIR)/"
	scp "$(REMOTE_HOST):$(OUTPUT_DIR)/$(DIAGRAM2)" "$(LOCAL_DIR)/"

# Download everything (logs + diagrams)
download: logs diagrams
