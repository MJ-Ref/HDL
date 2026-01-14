"""Tests for communication channels."""

import pytest
from lpca.channels.text import (
    NoCommChannel,
    FullTextChannel,
    BudgetedTextChannel,
    create_channel,
)
from lpca.channels.base import Message


class TestNoCommChannel:
    """Tests for P0: No Communication channel."""

    def test_create(self):
        channel = NoCommChannel()
        assert channel.name == "P0"
        assert channel.channel_type == "none"

    def test_encode_returns_empty(self):
        channel = NoCommChannel()
        msg = channel.encode("test message", "A", 0)
        assert msg.content is None
        assert msg.bits == 0

    def test_decode_returns_empty(self):
        channel = NoCommChannel()
        msg = channel.encode("test", "A", 0)
        result = channel.decode(msg)
        assert result == ""

    def test_send_receive_cycle(self):
        channel = NoCommChannel()
        channel.reset()
        msg = channel.send("hello", "A", "B", 0)
        received = channel.receive(msg)
        assert received == ""
        stats = channel.get_stats()
        assert stats["total_bits"] == 0


class TestFullTextChannel:
    """Tests for P1: Full Text channel."""

    def test_create(self):
        channel = FullTextChannel()
        assert channel.name == "P1"
        assert channel.channel_type == "text"

    def test_encode_message(self):
        channel = FullTextChannel()
        msg = channel.encode("hello world", "A", 0)
        assert msg.content == "hello world"
        assert msg.bits == len("hello world".encode("utf-8")) * 8

    def test_decode_message(self):
        channel = FullTextChannel()
        msg = channel.encode("test", "A", 0)
        result = channel.decode(msg)
        assert result == "test"

    def test_truncates_long_messages(self):
        channel = FullTextChannel(max_tokens=10)
        long_message = "x" * 1000
        msg = channel.encode(long_message, "A", 0)
        assert len(msg.content) <= 10 * 4  # ~4 chars per token

    def test_send_receive_tracks_bits(self):
        channel = FullTextChannel()
        channel.reset()
        channel.send("hello", "A", "B", 0)
        stats = channel.get_stats()
        assert stats["total_bits"] > 0
        assert stats["message_count"] == 1


class TestBudgetedTextChannel:
    """Tests for P2: Budgeted Text channel."""

    def test_create_with_budget(self):
        channel = BudgetedTextChannel(max_bytes=100)
        assert channel.name == "P2"
        assert channel.max_bytes == 100

    def test_truncates_to_budget(self):
        channel = BudgetedTextChannel(max_bytes=10)
        msg = channel.encode("this is a very long message", "A", 0)
        assert len(msg.content.encode("utf-8")) <= 10

    def test_tracks_budget_usage(self):
        channel = BudgetedTextChannel(max_bytes=100)
        channel.reset()
        channel.send("hello", "A", "B", 0)
        stats = channel.get_stats()
        assert stats["total_bits"] > 0
        assert stats["message_count"] == 1


class TestCreateChannel:
    """Tests for channel factory function."""

    def test_create_p0(self):
        channel = create_channel("P0")
        assert isinstance(channel, NoCommChannel)

    def test_create_p1(self):
        channel = create_channel("P1")
        assert isinstance(channel, FullTextChannel)

    def test_create_p2(self):
        channel = create_channel("P2")
        assert isinstance(channel, BudgetedTextChannel)

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError):
            create_channel("UNKNOWN")


class TestChannelStats:
    """Tests for channel statistics tracking."""

    def test_reset_clears_stats(self):
        channel = FullTextChannel()
        channel.send("test", "A", "B", 0)
        channel.reset()
        stats = channel.get_stats()
        assert stats["total_bits"] == 0
        assert stats["message_count"] == 0

    def test_multiple_messages_accumulate(self):
        channel = FullTextChannel()
        channel.reset()
        channel.send("hello", "A", "B", 0)
        channel.send("world", "B", "A", 1)
        stats = channel.get_stats()
        assert stats["message_count"] == 2
