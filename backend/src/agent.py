import logging
import json
import os
import asyncio
from datetime import datetime
from typing import Annotated, Literal
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    tokenize,
    metrics,
    MetricsCollectedEvent,
    RunContext,
    function_tool,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# ORDER MANAGEMENT SYSTEM
# ======================================================
@dataclass
class OrderState:
    """Coffee shop order state"""
    drinkType: str | None = None
    size: str | None = None
    milk: str | None = None
    extras: list[str] = field(default_factory=list)
    name: str | None = None

    def is_complete(self) -> bool:
        """Check if all required fields are filled"""
        return all([
            self.drinkType is not None,
            self.size is not None,
            self.milk is not None,
            self.name is not None,
        ])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "drinkType": self.drinkType,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name,
        }

    def get_summary(self) -> str:
        """Get readable order summary"""
        if not self.is_complete():
            return "Order in progress."

        extras_text = f" with {', '.join(self.extras)}" if self.extras else ""
        return f"{self.size.upper()} {self.drinkType.title()} with {self.milk.title()} milk{extras_text} for {self.name}"


@dataclass
class Userdata:
    """User session data"""
    order: OrderState
    session_start: datetime = field(default_factory=datetime.now)

# ======================================================
# BARISTA AGENT FUNCTION TOOLS
# ======================================================

@function_tool
async def set_drink_type(
    ctx: RunContext[Userdata],
    drink: Annotated[
        Literal["latte", "cappuccino", "americano", "espresso", "mocha", "coffee", "cold brew", "matcha"],
        Field(description="The type of coffee drink the customer wants"),
    ],
) -> str:
    """Set the drink type. Call when customer specifies which coffee they want."""
    ctx.userdata.order.drinkType = drink
    logger.info(f"Drink set: {drink}")
    logger.info(f"Order progress: {ctx.userdata.order.get_summary()}")
    return f"Got it. One {drink}."


@function_tool
async def set_size(
    ctx: RunContext[Userdata],
    size: Annotated[
        Literal["small", "medium", "large", "extra large"],
        Field(description="The size of the drink"),
    ],
) -> str:
    """Set the size. Call when customer specifies drink size."""
    ctx.userdata.order.size = size
    logger.info(f"Size set: {size}")
    logger.info(f"Order progress: {ctx.userdata.order.get_summary()}")
    return f"{size.title()} size for your {ctx.userdata.order.drinkType}."


@function_tool
async def set_milk(
    ctx: RunContext[Userdata],
    milk: Annotated[
        Literal["whole", "skim", "almond", "oat", "soy", "coconut", "none"],
        Field(description="The type of milk for the drink"),
    ],
) -> str:
    """Set milk preference. Call when customer specifies milk type."""
    ctx.userdata.order.milk = milk
    logger.info(f"Milk set: {milk}")
    logger.info(f"Order progress: {ctx.userdata.order.get_summary()}")

    if milk == "none":
        return "Got it, black coffee."
    return f"{milk.title()} milk noted."


@function_tool
async def set_extras(
    ctx: RunContext[Userdata],
    extras: Annotated[
        list[Literal["sugar", "whipped cream", "caramel", "extra shot", "vanilla", "cinnamon", "honey"]] | None,
        Field(description="List of extras, or empty/None for no extras"),
    ] = None,
) -> str:
    """Set extras. Call when customer specifies add-ons or says no extras."""
    ctx.userdata.order.extras = extras if extras else []
    logger.info(f"Extras set: {ctx.userdata.order.extras}")
    logger.info(f"Order progress: {ctx.userdata.order.get_summary()}")

    if ctx.userdata.order.extras:
        return f"Added {', '.join(ctx.userdata.order.extras)}."
    return "No extras."


@function_tool
async def set_name(
    ctx: RunContext[Userdata],
    name: Annotated[str, Field(description="Customer's name for the order")],
) -> str:
    """Set customer name. Call when customer provides their name."""
    ctx.userdata.order.name = name.strip().title()
    logger.info(f"Name set: {ctx.userdata.order.name}")
    logger.info(f"Order progress: {ctx.userdata.order.get_summary()}")
    return f"Thanks, {ctx.userdata.order.name}."


@function_tool
async def complete_order(ctx: RunContext[Userdata]) -> str:
    """Finalize and save order to JSON. Only call when all fields are filled."""
    order = ctx.userdata.order

    if not order.is_complete():
        missing = []
        if not order.drinkType:
            missing.append("drink type")
        if not order.size:
            missing.append("size")
        if not order.milk:
            missing.append("milk")
        if not order.name:
            missing.append("name")

        logger.warning(f"Cannot complete order. Missing: {', '.join(missing)}")
        return f"We're almost done. Please provide: {', '.join(missing)}."

    logger.info(f"Order ready for completion: {order.get_summary()}")

    try:
        save_order_to_json(order)
        extras_text = f" with {', '.join(order.extras)}" if order.extras else ""
        logger.info("Order completed successfully.")
        return (
            f"Your {order.size} {order.drinkType} with {order.milk} milk{extras_text} "
            f"is confirmed, {order.name}. We’ll start preparing it now."
        )

    except Exception as e:
        logger.error(f"Order save failed: {e}")
        return "Your order has been recorded, but there was an issue saving it. We'll still prepare your drink."


@function_tool
async def get_order_status(ctx: RunContext[Userdata]) -> str:
    """Get current order status. Call when customer asks about their order."""
    order = ctx.userdata.order
    if order.is_complete():
        return f"Your order is complete: {order.get_summary()}"

    progress = order.get_summary()
    return f"Order in progress: {progress}"

# ======================================================
# AGENT DEFINITION
# ======================================================

class BaristaAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly and professional barista cafe.

Your goal is to take coffee orders by collecting, step by step:
- Drink Type: latte, cappuccino, americano, espresso, mocha, coffee, cold brew, matcha
- Size: small, medium, large, extra large
- Milk: whole, skim, almond, oat, soy, coconut, none
- Extras: sugar, whipped cream, caramel, extra shot, vanilla, cinnamon, honey, or none
- Customer Name: for the order label

Process:
1. Greet and ask for the drink type.
2. Ask for size.
3. Ask for milk.
4. Ask about extras.
5. Ask for the customer's name.
6. Confirm the full order and complete it.

Style:
- Be polite, clear, and concise.
- Ask one question at a time.
- Confirm choices as you go.

Use the function tools to record each piece of information.
            """,
            tools=[
                set_drink_type,
                set_size,
                set_milk,
                set_extras,
                set_name,
                complete_order,
                get_order_status,
            ],
        )


def create_empty_order() -> OrderState:
    """Create a fresh order state."""
    return OrderState()

# ======================================================
# ORDER STORAGE & PERSISTENCE
# ======================================================

def get_orders_folder() -> str:
    """Get the orders directory path."""
    base_dir = os.path.dirname(__file__)   # src/
    backend_dir = os.path.abspath(os.path.join(base_dir, ".."))
    folder = os.path.join(backend_dir, "orders")
    os.makedirs(folder, exist_ok=True)
    return folder


def save_order_to_json(order: OrderState) -> str:
    """Save order to JSON file."""
    folder = get_orders_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"order_{timestamp}.json"
    path = os.path.join(folder, filename)

    try:
        order_data = order.to_dict()
        order_data["timestamp"] = datetime.now().isoformat()
        order_data["session_id"] = f"session_{timestamp}"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(order_data, f, indent=4, ensure_ascii=False)

        logger.info(f"Order saved to {path}")
        logger.info(f"Customer: {order.name}")
        logger.info(f"Order summary: {order.get_summary()}")

        return path

    except Exception as e:
        logger.error(f"Error saving order: {e}. Path attempted: {path}")
        raise e

# ======================================================
# SYSTEM INITIALIZATION & PREWARMING
# ======================================================

def prewarm(proc: JobProcess):
    """Preload VAD model for better performance."""
    logger.info("Prewarming VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded successfully.")

# ======================================================
# AGENT SESSION MANAGEMENT
# ======================================================

async def entrypoint(ctx: JobContext):
    """Main agent entrypoint - handles customer sessions."""
    ctx.log_context_fields = {"room": ctx.room.name}

    # Create user session data with empty order
    userdata = Userdata(order=create_empty_order())

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"New customer session: {session_id}")
    logger.info(f"Initial order state: {userdata.order.get_summary()}")

    # Create session with userdata
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    await session.start(
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

# ======================================================
# APPLICATION BOOTSTRAP & LAUNCH
# ======================================================

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
