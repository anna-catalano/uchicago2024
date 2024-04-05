from typing import Optional

from xchangelib import xchange_client
import asyncio
import statistics
import math

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

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
        
        count = 0
        # shorting the ETF's
        if count == 0:
            market_order_id = await self.place_order("JAK",5, xchange_client.Side.SELL)
            print("Market Order ID:", market_order_id)
            await asyncio.sleep(5)
            
            market_order_id = await self.place_order("SCP",5, xchange_client.Side.SELL)
            print("Market Order ID:", market_order_id)
            await asyncio.sleep(5)
            count += 1

        # 

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

    async def view_books(self):
        """Prints the books every 3 seconds with weighted averages.
            Books include (price, amount, standard deviation) for each order"""
        while True:
            await asyncio.sleep(3)
            weighted_avg_dict = {}

            for security, book in self.order_books.items():
                all_prices = [k for k, v in book.bids.items() if v != 0] + [k for k, v in book.asks.items() if v != 0]
                if not all_prices:
                    continue
                
                # Calculate mean of k
                mean_k = sum(all_prices) / len(all_prices)
                
                # Calculate standard deviation of k
                variance = sum((k - mean_k) ** 2 for k in all_prices) / len(all_prices)
                std_dev = math.sqrt(variance)
                
                # Adjust bids and asks by adding standard deviation from mean to each (k, v) tuple
                adjusted_bids = [(k, v, (k - mean_k) / std_dev if std_dev else 0) for k, v in book.bids.items() if v != 0]
                adjusted_asks = [(k, v, (k - mean_k) / std_dev if std_dev else 0) for k, v in book.asks.items() if v != 0]

                weighted_avg_dict[security] = self.calc_weighted_averages(adjusted_bids, adjusted_asks)  # returns a tuple
                print("mean:", mean_k)
                print(f"bids for {security}", adjusted_bids)
                print(f"asks for {security}", adjusted_asks)
                
            print("Weighted averages:", weighted_avg_dict)
    
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


