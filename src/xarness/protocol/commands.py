"""Write models: commands, their flat result, and the PATCH settings body.

Commands are the only way to mutate a session (the HTTP handlers route
them through the session's serialized ``execute``). ``CommandResult`` is
one flat model — the fields a given command sets are the ones its caller
reads — instead of a payload union per command.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from ..images import ImageRef
from .views import Mode, RefInfo


class CommandBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SendMessage(CommandBase):
    type: Literal["send_message"] = "send_message"
    text: str
    images: list[ImageRef] | None = None


class Cancel(CommandBase):
    type: Literal["cancel"] = "cancel"


class AnswerAsk(CommandBase):
    type: Literal["answer_ask"] = "answer_ask"
    ask_id: str
    answers: list[str] | None = None


class ResolveDrift(CommandBase):
    type: Literal["resolve_drift"] = "resolve_drift"
    accept: bool


class DiscardQueued(CommandBase):
    type: Literal["discard_queued"] = "discard_queued"


class FlushQueued(CommandBase):
    type: Literal["flush_queued"] = "flush_queued"


class Rollback(CommandBase):
    """Undo or retry. Without ``message_id``: the last user turn (resend
    re-sends it). With one: drop that message and everything after it
    (click-to-undo); ``resend`` is ignored."""

    type: Literal["rollback"] = "rollback"
    resend: bool = False
    message_id: str | None = None


class Compact(CommandBase):
    type: Literal["compact"] = "compact"


class Accept(CommandBase):
    type: Literal["accept"] = "accept"


class Reject(CommandBase):
    type: Literal["reject"] = "reject"


class SetMode(CommandBase):
    type: Literal["set_mode"] = "set_mode"
    mode: Mode


class SwitchModel(CommandBase):
    type: Literal["switch_model"] = "switch_model"
    name: str
    persist_default: bool = False


class SetSandbox(CommandBase):
    type: Literal["set_sandbox"] = "set_sandbox"
    name: str


class Filter(CommandBase):
    """Add/remove a sandbox mount exclusion."""

    type: Literal["filter"] = "filter"
    path: str
    add: bool


class Ref(CommandBase):
    """Add/remove an external reference path for the session's workspace."""

    type: Literal["ref"] = "ref"
    path: str
    add: bool


Command = Annotated[
    Union[
        SendMessage,
        Cancel,
        AnswerAsk,
        ResolveDrift,
        DiscardQueued,
        FlushQueued,
        Rollback,
        Compact,
        Accept,
        Reject,
        SetMode,
        SwitchModel,
        SetSandbox,
        Filter,
        Ref,
    ],
    Field(discriminator="type"),
]


class CommandResult(BaseModel):
    """Reply to one command, delivered only to its issuer.

    One flat model instead of a payload union per command: the fields a
    given command sets are the ones its caller reads."""

    cmd_id: str = ""
    ok: bool
    error: str | None = None
    # send_message
    started: bool | None = None
    queued: bool | None = None
    # cancel
    cancelled: bool | None = None
    # discard_queued
    discarded: int | None = None
    text: str | None = None
    # flush_queued
    flushed: bool | None = None
    # rollback
    user_text: str | None = None
    images: list[ImageRef] | None = None
    # settings commands
    mode: Mode | None = None
    model: str | None = None
    sandbox: str | None = None
    filter_list: list[str] | None = None
    refs: list[RefInfo] | None = None


class CommandEnvelope(BaseModel):
    """POST /sessions/{name}/commands body: one command + a client-generated id."""

    cmd_id: str
    command: Command


class SessionUpdate(BaseModel):
    """PATCH /sessions/{name}: the settings commands, on one typed body.

    A cleaner fit than five command types for slow-changing settings; the
    equivalent commands remain available for clients that prefer them."""

    mode: Mode | None = None
    model: str | None = None
    sandbox: str | None = None
    filter_path: str | None = None
    filter_add: bool | None = None
    ref_path: str | None = None
    ref_add: bool | None = None


class DeleteResult(BaseModel):
    deleted: bool


class CreateSessionRequest(BaseModel):
    """POST /v1/sessions body.

    ``sandbox`` names a sandbox profile; ``use_sandbox``/``network``/``gpu``/
    ``enable_tracking`` are the tri-state CLI overrides (None = keep the
    profile's value, which is also what an absent flag sends)."""

    workspace: str | None = None
    model: str | None = None
    sandbox: str | None = None
    preset: str | None = None
    no_init_repo: bool = False
    use_sandbox: bool | None = None
    network: bool | None = None
    gpu: bool | None = None
    enable_tracking: bool | None = None
    no_session: bool = False
    first_message: str | None = None
