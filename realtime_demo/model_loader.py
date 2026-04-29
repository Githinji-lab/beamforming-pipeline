def load_realtime_bundle(
    project_root: str,
    model_path: Optional[str] = None,
    artifacts_path: Optional[str] = None,
    tflite_model_path: Optional[str] = None,
    inference_backend: str = "tflite",
) -> RealtimeBeamModel:
    _ensure_src_on_path(project_root)

    # --- MONKEY PATCH START ---
    # We override the internal Dense __init__ to strip 'quantization_config'
    # This bypasses the Keras 3 deserialization bug globally.
    import tensorflow as tf
    from tensorflow.keras.layers import Dense
    
    original_init = Dense.__init__
    
    def patched_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        return original_init(self, *args, **kwargs)
    
    Dense.__init__ = patched_init
    # --- MONKEY PATCH END ---

    from simulators import BeamformingSimulatorV4
    import dqn_beam_agent
    _ = dqn_beam_agent

    results_dir = os.path.join(project_root, "results")
    resolved_model_path = model_path or os.path.join(results_dir, "dqn_beam_model.keras")
    resolved_tflite_path = tflite_model_path or os.path.join(results_dir, "dqn_beam_model_int8.tflite")
    resolved_artifacts_path = artifacts_path or os.path.join(results_dir, "dqn_beam_artifacts.pkl")

    inference_backend = str(inference_backend).lower().strip()
    if inference_backend not in {"keras", "tflite"}:
        raise ValueError("inference_backend must be either 'keras' or 'tflite'")

    if not os.path.exists(resolved_artifacts_path):
        raise FileNotFoundError(f"Artifacts not found: {resolved_artifacts_path}")

    model = None
    interpreter = None

    if inference_backend == "tflite":
        if not os.path.exists(resolved_tflite_path):
            raise FileNotFoundError(f"TFLite model not found: {resolved_tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=resolved_tflite_path)
        interpreter.allocate_tensors()
    else:
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"Model not found: {resolved_model_path}")
        
        # Now we can load normally; the monkey patch will catch the error
        model = tf.keras.models.load_model(resolved_model_path, compile=False)

    with open(resolved_artifacts_path, "rb") as f:
        artifacts = pickle.load(f)

    simulator = BeamformingSimulatorV4(N_tx=8, K=4)
    default_snr_db = float(simulator.snr_db_list[len(simulator.snr_db_list) // 2])

    return RealtimeBeamModel(
        project_root=project_root,
        model=model,
        tflite_interpreter=interpreter,
        inference_backend=inference_backend,
        codebook=artifacts["codebook"],
        state_encoder=artifacts["state_encoder"],
        state_scaler=artifacts["state_scaler"],
        phase1_augmenter=artifacts.get("phase1_augmenter", None),
        simulator=simulator,
        default_snr_db=default_snr_db,
    )
