import time
from dataclasses import dataclass
from typing import Optional, Callable

@dataclass
class Progress:
    total: int
    print_every: int = 25              # print cadence in steps
    alpha: float = 0.2                 # EMA smoothing of step time
    unit_name: str = "steps"           # throughput unit
    logger: Optional[Callable[[str], None]] = None
    only_rank0: bool = True            # print only on rank 0
    debug: bool = False                # set True to debug suppression logic

    # internal state (do not pass)
    _t0: float = 0.0
    _ema_step_s: Optional[float] = None
    _last_print_step: int = 0          # important: start at 0 for 1-based loops

    def start(self, current_step: int = 1) -> None:
        """Call once before the loop (1-based indexing)."""
        self._t0 = time.perf_counter()
        self._ema_step_s = None
        self._last_print_step = 0      # keep 0 so step==print_every prints

    def update(self, step: int, batch_size: Optional[int] = None,
               rank: int = 0, should_print: Optional[bool] = None) -> None:
        """
        step: 1-based step index.
        should_print: True=force print now; False/None=use cadence.
        """
        if self.only_rank0 and rank != 0:
            return

        force = (should_print is True)  # only True forces; False/None -> cadence

        # throttle unless forced; always allow final step
        if not force:
            if step != self.total:
                if step - self._last_print_step < self.print_every:
                    if self.debug and step == 1:
                        (self.logger or print)("[ProgressETA] suppressed at step 1 (warming up)")
                    return

        now = time.perf_counter()
        steps_done = max(1, step)  # 1-based

        elapsed = now - self._t0
        mean_step = elapsed / steps_done

        # EMA of step time
        self._ema_step_s = mean_step if self._ema_step_s is None else \
            (1 - self.alpha) * self._ema_step_s + self.alpha * mean_step

        remaining_steps = max(0, self.total - steps_done)
        eta_sec = remaining_steps * self._ema_step_s

        # optional throughput
        msg = (f"[step {steps_done}/{self.total}] "
               f"elapsed {elapsed:,.1f}s | ETA {eta_sec:,.1f}s")
        if batch_size is not None and elapsed > 0:
            thr = (steps_done * batch_size) / elapsed
            msg += f" | {thr:,.1f} {self.unit_name}/s"

        (self.logger or print)(msg)
        self._last_print_step = step

    def stop(self, rank: int = 0) -> None:
        if self.only_rank0 and rank != 0:
            return
        elapsed = time.perf_counter() - self._t0
        (self.logger or print)(f"[done] total elapsed {elapsed:,.1f}s")
