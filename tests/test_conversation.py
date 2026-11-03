"""Domain invariants for the conversation model."""

from xarness.conversation import Conversation, user_message


def test_sent_user_message_owns_a_turn_checkpoint() -> None:
    conversation = Conversation()
    user = conversation.add(user_message("hi"))
    conversation.begin_turn(user, "before")

    turn = conversation.turns[user.id]
    assert turn.before_tree == "before"
    assert turn.after_tree is None  # open
    assert conversation.open_turn is turn
    # The message itself stays immutable: the checkpoint lives in the
    # conversation's side table, keyed by the message id.
    assert "turn" not in user.model_dump()


def test_steered_user_message_does_not_own_a_checkpoint() -> None:
    conversation = Conversation()
    user = conversation.add(user_message("hi"))
    conversation.begin_turn(user, "before")
    steered = conversation.add(user_message("steered"))

    assert user.id in conversation.turns
    assert steered.id not in conversation.turns


def test_open_and_closed_turn() -> None:
    conversation = Conversation()
    user = conversation.add(user_message("hi"))
    conversation.begin_turn(user, "before")
    turn = conversation.turns[user.id]
    assert conversation.open_turn is turn

    turn.closed = True
    assert conversation.open_turn is None


def test_lock_turns_leaves_only_the_open_turn_unlocked() -> None:
    conversation = Conversation()
    first = conversation.add(user_message("one"))
    conversation.begin_turn(first, "a")
    conversation.turns[first.id].after_tree = "b"
    second = conversation.add(user_message("two"))
    conversation.begin_turn(second, "b")

    conversation.lock_turns("c")

    first_turn = conversation.turns[first.id]
    second_turn = conversation.turns[second.id]
    assert first_turn.before_tree == first_turn.after_tree == "c"
    assert second_turn.before_tree == "c"
    assert second_turn.after_tree is None  # still open


def test_rollback_prunes_dropped_turns() -> None:
    conversation = Conversation()
    first = conversation.add(user_message("one"))
    conversation.begin_turn(first, "a")
    second = conversation.add(user_message("two"))
    conversation.begin_turn(second, "b")

    plan = conversation.rollback_plan_to(first)
    conversation.apply_rollback(plan)

    # Rolling back TO first truncates at first: both turns are gone.
    assert second.id not in conversation.turns
    assert first.id not in conversation.turns
    assert conversation.turns == {}


def test_the_last_summary_defines_the_visible_slice() -> None:
    conversation = Conversation()
    conversation.add(user_message("old"))
    conversation.add(user_message("summary", is_summary=True))

    assert [m.content for m in conversation.visible_messages()] == ["summary"]
    assert [m.content for m in conversation.history_messages()] == ["old"]
    # Undoing a compaction pops the summary: the old messages come back.
    conversation.apply_rollback(
        conversation.rollback_plan()
    )
    assert [m.content for m in conversation.visible_messages()] == ["old"]
