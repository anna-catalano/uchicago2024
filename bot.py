from typing import Optional

from xchangelib import xchange_client
import asyncio
import statistics
import math

class MyXchangeClient(xchange_client.XChangeClient):

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.weighted_avg_dict = {}
        self.adjusted_order_books = {}

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        await asyncio.sleep(5)
        
        # # example code
        
        # print("attempting to trade")
        # await self.place_order("BRV",3, xchange_client.Side.SELL, 7)

        # # Cancelling an order
        # order_to_cancel = await self.place_order("BRV",3, xchange_client.Side.BUY, 5)
        # await asyncio.sleep(5)
        # await self.cancel_order(order_to_cancel)

        # # Placing Swap requests
        # await self.place_swap_order('toJAK', 1)
        # await asyncio.sleep(5)
        # await self.place_swap_order('fromSCP', 1)
        # await asyncio.sleep(5)

        # # Placing an order that gets rejected for exceeding qty limits
        # await self.place_order("BRV",1000, xchange_client.Side.SELL, 7)
        # await asyncio.sleep(5)

        # # Placing a market order
        # market_order_id = await self.place_order("BRV",10, xchange_client.Side.SELL)
        # print("Market Order ID:", market_order_id)
        # await asyncio.sleep(5)
        
        
        # shorting the ETFs
        # only short the two ETFs one time
        count = 0
        if count == 0:
            market_order_id = await self.place_order("JAK",5, xchange_client.Side.SELL)
            print("Market Order ID:", market_order_id)
            
            market_order_id = await self.place_order("SCP",5, xchange_client.Side.SELL)
            print("Market Order ID:", market_order_id)
            
            count += 1

        # calculating margins
        # print("SELF ADJUSTED BOOKS", self.adjusted_order_books)
        for security, books in self.adjusted_order_books.items(): # books includes bids, asks, mean
            lower_margin_count = 0
            upper_margin_count = 0
            
            for order in books['bids']:
                price, amt, stdev = order
                # if the standard deviation for the current bid exceeds 2, buy
                if stdev <= -1: 
                    market_order_id = await self.place_order(security, 3, xchange_client.Side.BUY)
                    print("Market Order ID:", market_order_id)
                    lower_margin_count += 1
            for order in books['asks']:
                price, amt, stdev = order
                if stdev >= 1: 
                    market_order_id = await self.place_order(security, 3, xchange_client.Side.SELL)
                    print("Market Order ID:", market_order_id)
                    upper_margin_count += 1
            
            # later implementation:
            # implement counter variables and update mean + stdev accordingly
            # if one counter exceeds 4, move mean to margin and recalc stdev
            # continuee down orders and keep calculating 
            # means/stdev are reset with next round

        # Viewing Positions
        print("My positions:", self.positions)

    def calc_weighted_averages(self, bids, asks):
        def weighted_average(prices):
            total = sum(price * freq for price, freq, _ in prices)
            weight_sum = sum(freq for _, freq, _ in prices)
            return total / weight_sum if weight_sum else 0

        # Calculate weighted averages for bids and asks
        bid_weighted_average = weighted_average(bids)
        ask_weighted_average = weighted_average(asks)

        return (bid_weighted_average, ask_weighted_average)

    def adjust_orders_with_stddev(self, orders, mean_k):
        """Adjust orders with standard deviation from the mean.
           Takes the current mean and the list of orders,
           returns the adjusted list with (price, amount, standard deviation)."""
        if not orders:
            return []

        # Calculate standard deviation
        variance = sum((price - mean_k) ** 2 for price, _, _ in orders) / len(orders)
        std_dev = math.sqrt(variance)

        # Adjust orders by adding standard deviation from mean to each (price, amount) tuple
        adjusted_orders = [(price, amount, (price - mean_k) / std_dev if std_dev else 0) for price, amount, _ in orders]
        return adjusted_orders

    async def view_books(self):
        """Prints the books every 3 seconds with weighted averages.
           Books include (price, amount, standard deviation) for each order"""
        while True:
            await asyncio.sleep(3)
            local_weighted_avg_dict = {}
            local_adjusted_order_books = {}

            for security, book in self.order_books.items(): # security is the key, book is a dict with bids and ask as keys, list as value
                all_prices = [k for k, v in book.bids.items() if v != 0] + [k for k, v in book.asks.items() if v != 0]
                if not all_prices:
                    continue

                # Calculate mean of k
                mean_k = sum(all_prices) / len(all_prices)

                # Extract and adjust bids and asks separately with standard deviation
                raw_bids = [(k, v, 0) for k, v in book.bids.items() if v != 0]  # Initial std dev is 0
                raw_asks = [(k, v, 0) for k, v in book.asks.items() if v != 0]

                adjusted_bids = self.adjust_orders_with_stddev(raw_bids, mean_k)
                adjusted_asks = self.adjust_orders_with_stddev(raw_asks, mean_k)

                # Update local dictionaries
                local_weighted_avg_dict[security] = self.calc_weighted_averages(adjusted_bids, adjusted_asks)
                local_adjusted_order_books[security] = {
                    'mean': mean_k,
                    'bids': adjusted_bids,
                    'asks': adjusted_asks
                }

            # Update class-level dictionaries with local copies
            self.weighted_avg_dict = local_weighted_avg_dict
            self.adjusted_order_books = local_adjusted_order_books

            # print("Weighted averages:", self.weighted_avg_dict)
            # for security, books in self.adjusted_order_books.items():
            #     print(f"{security} bids:", books['bids'])
            #     print(f"{security} asks:", books['asks'])
    
    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        asyncio.create_task(self.trade())
        asyncio.create_task(self.view_books())
        await self.connect()


async def main():
    SERVER = '18.188.190.235:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER,"nyu","graveler-dodrio-8874")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())


