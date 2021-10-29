reverse_id_mapping = {
    v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
}
for result in self._coco_results:
    result["category_id"] = reverse_id_mapping[result["category_id"]]