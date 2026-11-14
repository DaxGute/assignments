"""Sanity-check H100 availability."""
if __name__ == "__main__":
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets

    @app.function(
        image=build_image(), volumes=VOLUME_MOUNTS, gpu="H100", timeout=600,
        secrets=secrets(),
    )
    def ping():
        import torch
        return {
            "ok": True,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        print(ping.remote())
