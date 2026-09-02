"""Single-child foreground execution with exact signal forwarding."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from pathlib import Path
from types import FrameType
from typing import Callable, Mapping, MutableMapping, Sequence

from posttrain_circuits.scheduler_adapter.errors import DispatchError


PopenFactory = Callable[..., subprocess.Popen[str]]


def run_foreground_child(
    argv: Sequence[str],
    *,
    cwd: Path,
    environ: Mapping[str, str] | None = None,
    pass_fds: tuple[int, ...] = (),
    popen: PopenFactory = subprocess.Popen,
) -> int:
    """Run and synchronously wait for exactly one code-owned child process."""

    command = tuple(argv)
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise DispatchError("fixed handler argv is invalid")
    working_directory = Path(cwd)
    child_environment: MutableMapping[str, str] = dict(
        os.environ if environ is None else environ
    )
    forwarded_signals = tuple(
        signum for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    )
    if any(
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 0
        for descriptor in pass_fds
    ) or len(set(pass_fds)) != len(pass_fds):
        raise DispatchError("pass_fds must contain unique open descriptors")
    if threading.active_count() != 1:
        raise DispatchError("foreground adapter must remain single-threaded before exec")
    try:
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, forwarded_signals)
    except (AttributeError, OSError, ValueError) as error:
        raise DispatchError(f"cannot block forwarding signals before spawn: {error}") from error
    if set(previous_mask) & set(forwarded_signals):
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise DispatchError(
            "foreground adapter cannot start with forwarding signals already blocked"
        )
    try:
        try:
            def restore_child_signal_mask() -> None:
                # Popen's forked child inherits the parent's temporary block.
                # This entrypoint is deliberately single-child/single-threaded;
                # restore the pre-block mask immediately before exec so an
                # exactly-forwarded signal can terminate the real handler.
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

            child = popen(
                command,
                cwd=working_directory,
                env=child_environment,
                shell=False,
                start_new_session=False,
                close_fds=True,
                pass_fds=pass_fds,
                preexec_fn=restore_child_signal_mask,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise DispatchError(f"cannot launch fixed OPD handler: {error}") from error
    except BaseException:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        raise

    previous: dict[signal.Signals, signal.Handlers] = {}
    pending: list[int] = []

    def forward(signum: int, _frame: FrameType | None) -> None:
        pending.append(signum)
        if child.poll() is None:
            try:
                # Forward only to the exact direct child. Never signal a group.
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    unblocked = False
    try:
        for signum in forwarded_signals:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, forward)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        unblocked = True
        # Preserve Popen's exact status: negative means the child really died
        # from that signal; a handler-chosen positive status remains positive.
        return int(child.wait())
    except BaseException:
        if child.poll() is None:
            try:
                child.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            child.wait()
        raise
    finally:
        if unblocked:
            signal.pthread_sigmask(signal.SIG_BLOCK, forwarded_signals)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def propagate_child_signal(return_code: int) -> int:
    """Terminate this entrypoint with the same signal as its direct child."""

    if return_code >= 0:
        return return_code
    signum = -return_code
    if signum not in signal.valid_signals():
        raise DispatchError(f"child returned unsupported signal status: {return_code}")
    if signum not in {signal.SIGKILL, signal.SIGSTOP}:
        signal.signal(signum, signal.SIG_DFL)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signum,))
    os.kill(os.getpid(), signum)
    # Only reached under a mocked/abnormal signal implementation.
    return 128 + signum
