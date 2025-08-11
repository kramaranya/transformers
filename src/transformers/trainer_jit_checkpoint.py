import os
import signal
import threading
from typing import Optional
import torch
from .trainer_callback import TrainerCallback
from .utils import logging


logger = logging.get_logger(__name__)


class CheckpointManager:
    def __init__(self, trainer, grace_period: int = 30):
        self.trainer = trainer
        self.grace_period = grace_period
        self.checkpoint_thread = None
        self.checkpoint_stream = None
        self.checkpoint_requested = False
        self.checkpoint_in_progress = False
        self._original_sigterm_handler = None

        if torch.cuda.is_available():
            self.checkpoint_stream = torch.cuda.Stream()

    def setup_signal_handler(self):
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._sigterm_handler)
        logger.info("JIT checkpoint signal handler registered for SIGTERM")

    def _sigterm_handler(self, signum, frame):
        if self.checkpoint_requested:
            return
        logger.info(f"SIGTERM received, initiating JIT checkpoint with {self.grace_period}s grace period")
        self.checkpoint_requested = True
        # Start immediate checkpoint in a separate thread
        self.checkpoint_thread = threading.Thread(
            target=self._immediate_async_checkpoint,
            daemon=True
        )
        self.checkpoint_thread.start()

    def _immediate_async_checkpoint(self):
        """Immediate checkpoint using CUDA streams to avoid blocking training"""
        try:
            logger.info("Starting immediate async JIT checkpoint")

            # Capture the current stream before switching
            current_stream = torch.cuda.current_stream()

            # Wait for current CUDA operations to complete
            current_stream.wait_stream(self.checkpoint_stream)

            # Switch to checkpoint stream for all checkpoint operations
            with torch.cuda.stream(self.checkpoint_stream):
                self._save_jit_checkpoint()

            # Synchronize checkpoint stream
            self.checkpoint_stream.synchronize()

            # Switch back to the original stream
            with torch.cuda.stream(current_stream):
                pass

            logger.info("Immediate async JIT checkpoint completed successfully")

        except Exception as e:
            logger.error(f"Failed to complete immediate async JIT checkpoint: {e}")

    def execute_jit_checkpoint(self):
        if self.checkpoint_in_progress:
            logger.warning("Checkpoint already in progress, skipping")
            return

        self.checkpoint_in_progress = True

        try:
            logger.info("Starting JIT checkpoint save")
            self._save_jit_checkpoint()
            logger.info("JIT checkpoint completed successfully")
        except Exception as e:
            logger.error(f"Failed to complete JIT checkpoint: {e}")
            raise
        finally:
            self.checkpoint_in_progress = False
            if hasattr(self.trainer, 'control'):
                self.trainer.control.should_training_stop = True

    def _save_jit_checkpoint(self):
        try:
            original_step = self.trainer.state.global_step
            logger.info(f"Saving JIT checkpoint at step {original_step}")

            # Ensure we're on the checkpoint stream
            assert torch.cuda.current_stream() == self.checkpoint_stream

            # For FSDP models, temporarily set state dict type to avoid FORWARD state issues
            if hasattr(self.trainer.model, 'fsdp_wrapped_module'):
                original_state_dict_type = None
                try:
                    # Try to get current state dict type
                    if hasattr(self.trainer.model, 'get_state_dict_type'):
                        original_state_dict_type = self.trainer.model.get_state_dict_type()
                        self.trainer.model.set_state_dict_type('LOCAL_STATE_DICT')
                    elif hasattr(self.trainer.model, 'state_dict_type'):
                        original_state_dict_type = self.trainer.model.state_dict_type()
                        self.trainer.model.state_dict_type('LOCAL_STATE_DICT')
                except Exception as e:
                    logger.warning(f"Could not set FSDP state dict type: {e}")


            # Call the trainer's checkpoint method directly
            self.trainer._save_checkpoint(self.trainer.model, trial=None)
            self.checkpoint_requested = False

            if self.trainer.is_world_process_zero():
                run_dir = self.trainer._get_output_dir(trial=None)
                regular_checkpoint_dir = os.path.join(run_dir, f"checkpoint-{original_step}")
                jit_checkpoint_dir = os.path.join(run_dir, f"checkpoint-jit-{original_step}")

                if os.path.exists(regular_checkpoint_dir):
                    os.rename(regular_checkpoint_dir, jit_checkpoint_dir)
                    logger.info(f"JIT checkpoint saved to {jit_checkpoint_dir}")
                else:
                    logger.warning(f"Expected checkpoint directory not found: {regular_checkpoint_dir}")

        except Exception as e:
            logger.error(f"Failed to save JIT checkpoint: {e}")
            raise

    def should_checkpoint_now(self) -> bool:
        return self.checkpoint_requested


class JITCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.trainer = None
        self.jit_manager: Optional[CheckpointManager] = None

    def set_trainer(self, trainer):
        self.trainer = trainer
        if trainer.args.jit_checkpoint_on_sigterm:
            self.jit_manager = CheckpointManager(
                trainer=trainer,
                grace_period=trainer.args.jit_checkpoint_grace_period
            )
            self.jit_manager.setup_signal_handler()
            logger.info("JIT checkpointing enabled for Kubernetes/PyTorchJob environment")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            # try:
            #     self.jit_manager.execute_jit_checkpoint()
            # except Exception as e:
            #     logger.error(f"Error in pre-optimizer step checkpoint: {e}")
            # finally:
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.jit_manager and self.jit_manager.should_checkpoint_now():
            # try:
            #     logger.info("JIT checkpoint triggered at step end")
            #     self.jit_manager.execute_jit_checkpoint()
            # except Exception as e:
            #     logger.error(f"Error in step end checkpoint strategy: {e}")
            # finally:
            control.should_training_stop = True
