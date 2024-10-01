from pydantic import BaseModel


class Customer(BaseModel):
    CustomerID: int
    Age: int
    Gender: str
    Tenure: int
    UsageFrequency: int
    SupportCalls: int
    PaymentDelay: int
    SubscriptionType: str
    ContractLength: str
    TotalSpend: float
    LastInteraction: int
