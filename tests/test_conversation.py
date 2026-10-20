"""Domain invariants for the conversation model (Phase 2 guard rails)."""

from xarness.conversation import Conversation, Message


def test_sent_user_message_owns_a_turn_checkpoint() -> None:
    conversation = Conversation()
    user = conversation.add(Message(role="user", content="hi"))
    turn = conversation.begin_turn(user, "before")

    assert turn.user_message_id == user.id
    assert turn.before_tree == "before"
    assert turn.after_tree is None  # open
    assert conversation.open_turn is turn


def test_steered_user_message_does_not_own_a_checkpoint() -> None:
    conversation = Conversation()
    user = conversation.add(Message(role="user", content="hi"))
    conversation.begin_turn(user, "before")
    conversation.add(Message(role="user", content="steered"))

    assert len(conversation.turns) == 1
    assert conversation.turns[0].user_message_id == user.id


def test_open_and_closed_turn() -> None:
    conversation = Conversation()
    user = conversation.add(Message(role="user", content="hi"))
    turn = conversation.begin_turn(user, "before")
    assert conversation.open_turn is turn

    turn.after_tree = "after"
    assert conversation.open_turn is None


def test_lock_turns_leaves_only_the_open_turn_unlocked() -> None:
    conversation = Conversation()
    first = conversation.add(Message(role="user", content="one"))
    conversation.begin_turn(first, "a")
    conversation.turns[-1].after_tree = "b"
    second = conversation.add(Message(role="user", content="two"))
    conversation.begin_turn(second, "b")

    conversation.lock_turns("c")

    closed, opened = conversation.turns
    assert closed.locked and closed.before_tree == closed.after_tree == "c"
    assert not opened.locked and opened.before_tree == "c"
    assert opened.after_tree is None  # still open


def test_epochs_hide_compacted_history() -> None:
    conversation = Conversation()
    conversation.add(Message(role="user", content="old"))
    conversation.epoch += 1
    conversation.add(Message(role="user", content="summary", kind="summary"))

    assert [m.content for m in conversation.visible_messages()] == ["summary"]
    conversation.epoch -= 1
    assert [m.content for m in conversation.visible_messages()] == ["old"]
