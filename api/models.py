#schema layers of the api- api grammer
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class SurvivalPredictionRequest(BaseModel):
    Order_Quantity: float = Field(..., description="Quantity of the order")
    Order_Volume: float = Field(..., description="Volume of the order")
    Order_Weight: float = Field(..., description="Weight of the order")
    Priority_Flag: str = Field(..., description="Priority flag")
    Fulfiller_ID: str = Field(..., description="Fulfiller identifier")
    Routing_Lane_ID: str = Field(..., description="Routing lane identifier")
    Fulfiller_Throughput: float = Field(..., description="Fulfiller throughput rate")
    Total_Backlog_Ack: float = Field(..., description="Total backlog at acknowledgement")
    Current_Backlog: float = Field(..., description="Current backlog")
    Relative_Queue_Position: float = Field(..., description="Relative position in queue")
    Estimated_Processing_Rate: float = Field(..., description="Estimated processing rate")
    Days_in_Queue: float = Field(..., description="Days spent in queue")
    Day_of_Week: int = Field(..., description="Day of week (1-7)")
    Day_of_Month: int = Field(..., description="Day of month (1-31)")
    Month: int = Field(..., description="Month (1-12)")
    Season: str = Field(..., description="Season")
    Peak_Season: bool = Field(..., description="Is peak season")
    Demand_Surge: bool = Field(..., description="Is demand surge")
    Recent_Shipments: float = Field(..., description="Recent shipments count")
    Lead_Time_Trend: float = Field(..., description="Lead time trend")
    Geography: str = Field(..., description="Geographic location")
    Carrier: str = Field(..., description="Carrier name")
    Product_Category: str = Field(..., description="Product category")
    Order_Creation_DateTime: datetime = Field(..., description="Order creation timestamp")
    Acknowledgement_DateTime: datetime = Field(..., description="Acknowledgement timestamp")

class SurvivalPercentiles(BaseModel):
    p50: float = Field(..., description="50th percentile survival time")
    p90: float = Field(..., description="90th percentile survival time")
    mean: float = Field(..., description="Mean survival time")

class SurvivalProbabilityPoint(BaseModel):
    time: float = Field(..., description="Time point")
    probability: float = Field(..., description="Survival probability at this time")

class SurvivalPredictionResponse(BaseModel):
    percentiles: SurvivalPercentiles
    survival_curve: List[SurvivalProbabilityPoint]
    risk_score: float = Field(..., description="Relative risk score")
    event_probability: float = Field(..., description="Probability of event occurring")
    success: bool = Field(..., description="Prediction success status")
    message: str = Field(..., description="Additional message")

class BatchPredictionRequest(BaseModel):
    requests: List[SurvivalPredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[SurvivalPredictionResponse]
    processing_time: float
    total_processed: int