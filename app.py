import sys
import logging
from typing import Dict, Any, Optional, Union
from logging.config import dictConfig

from uvicorn import run as app_run
from fastapi.staticfiles import StaticFiles
from src.constants import APP_HOST, APP_PORT
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import Response, HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.pipeline.training_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import VehicleOwner, OwnerClassifier

# Configure logging
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "app.log",
            "level": "DEBUG",
            "formatter": "detailed",
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "fastapi": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False,
        },
        "app": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": False},
    },
    "root": {"handlers": ["console", "file"], "level": "DEBUG"},
}

dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("app")


class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized exception handling across the FastAPI application.

    This middleware catches all unhandled exceptions in the application and
    provides consistent error responses while logging detailed error information
    for debugging purposes.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process requests and handle any exceptions that occur during processing.

        Args:
            request: The incoming HTTP request object.
            call_next: The next middleware or route handler in the chain.

        Returns:
            Response: HTTP response object, either from successful processing
                or error handling.
        """
        try:
            logger.debug("Processing request: %s %s", request.method, request.url)
            response = await call_next(request)
            logger.debug(
                "Request completed successfully: %s %s", request.method, request.url
            )
            return response

        except HTTPException as e:
            logger.warning(
                "HTTP Exception occurred: %s - %s %s",
                e.status_code,
                request.method,
                request.url,
            )
            return JSONResponse(
                status_code=e.status_code,
                content={"error": e.detail, "status_code": e.status_code},
            )

        except Exception as e:
            logger.error(
                "Unhandled exception occurred: %s - %s %s",
                str(e),
                request.method,
                request.url,
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "status_code": 500},
            )


# Initialize FastAPI application
app = FastAPI(
    title="Vehicle Insurance Prediction API",
    description="ML-powered API for predicting vehicle insurance purchase likelihood",
    version="1.0.0",
)

# Add middleware
app.add_middleware(ExceptionHandlerMiddleware)

# Configure static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("FastAPI application initialized successfully")


class DataForm:
    """
    Handler class for processing HTML form data from vehicle insurance prediction requests.

    This class manages the extraction, validation, and conversion of form data
    into appropriate data types for machine learning model input. It handles
    both individual field conversions and special processing for categorical
    vehicle age data.

    Attributes:
        request (Request): The FastAPI request object containing form data.
        age (Optional[int]): Vehicle owner's age in years.
        gender (Optional[str]): Vehicle owner's gender ('Male' or 'Female').
        vintage (Optional[int]): Days since customer first associated with company.
        region_code (Optional[float]): Customer's region identifier code.
        annual_premium (Optional[float]): Annual insurance premium amount.
        vehicle_damage (Optional[str]): Past vehicle damage status ('Yes' or 'No').
        driving_license (Optional[int]): Valid driving license status (0 or 1).
        previously_insured (Optional[int]): Previous insurance status (0 or 1).
        policy_sales_channel (Optional[float]): Customer outreach channel code.
        vehicle_age_1_2_year (Optional[int]): Vehicle age 1-2 years indicator.
        vehicle_age_lt_1_year (Optional[int]): Vehicle age < 1 year indicator.
        vehicle_age_gt_2_years (Optional[int]): Vehicle age > 2 years indicator.
    """

    def __init__(self, request: Request) -> None:
        """
        Initialize DataForm with the request object.

        Args:
            request: FastAPI Request object containing the form data.
        """
        logger.debug("Initializing DataForm for request processing")

        self.request = request
        self.age: Optional[int] = None
        self.gender: Optional[str] = None
        self.vintage: Optional[int] = None
        self.region_code: Optional[float] = None
        self.annual_premium: Optional[float] = None
        self.vehicle_damage: Optional[str] = None
        self.driving_license: Optional[int] = None
        self.previously_insured: Optional[int] = None
        self.policy_sales_channel: Optional[float] = None
        self.vehicle_age_1_2_year: Optional[int] = None
        self.vehicle_age_lt_1_year: Optional[int] = None
        self.vehicle_age_gt_2_years: Optional[int] = None

    async def get_form_data(self) -> None:
        """
        Extract and process form data from the HTTP request.

        This method asynchronously retrieves form data from the request,
        validates and converts each field to appropriate data types, and
        processes special categorical fields like vehicle age categories.

        Raises:
            HTTPException: If form data extraction or validation fails with
                status code 422 (Unprocessable Entity).
        """
        try:
            logger.debug("Extracting form data from request")

            form = await self.request.form()

            # Extract and convert individual fields
            self.age = self._convert_to_int(form.get("Age"))
            self.gender = self._convert_to_object(form.get("Gender"))
            self.vintage = self._convert_to_int(form.get("Vintage"))
            self.region_code = self._convert_to_float(form.get("Region_Code"))
            self.annual_premium = self._convert_to_float(form.get("Annual_Premium"))
            self.vehicle_damage = self._convert_to_object(form.get("Vehicle_Damage"))
            self.driving_license = self._convert_to_int(form.get("Driving_License"))
            self.previously_insured = self._convert_to_int(
                form.get("Previously_Insured")
            )
            self.policy_sales_channel = self._convert_to_float(
                form.get("Policy_Sales_Channel")
            )

            # Process vehicle age category
            vehicle_age_category = form.get("Vehicle_Age_Category")
            self._process_vehicle_age_category(vehicle_age_category)

            logger.info(
                "Form data extracted successfully: age=%s, gender=%s, vehicle_damage=%s",
                self.age,
                self.gender,
                self.vehicle_damage,
            )

        except Exception as e:
            logger.error("Form validation error: %s", str(e), exc_info=True)
            raise HTTPException(
                status_code=422, detail=f"Form validation error: {str(e)}"
            )

    def _process_vehicle_age_category(self, category: Optional[str]) -> None:
        """
        Process vehicle age category selection into binary indicator variables.

        This method converts a single vehicle age category selection into
        three binary indicator variables for one-hot encoding.

        Args:
            category: Vehicle age category string ('1_2_year', 'lt_1_year', 'gt_2_years').
        """
        logger.debug("Processing vehicle age category: %s", category)

        # Initialize all age category indicators to 0
        self.vehicle_age_1_2_year = 0
        self.vehicle_age_lt_1_year = 0
        self.vehicle_age_gt_2_years = 0

        # Set appropriate indicator based on category
        if category == "1_2_year":
            self.vehicle_age_1_2_year = 1
        elif category == "lt_1_year":
            self.vehicle_age_lt_1_year = 1
        elif category == "gt_2_years":
            self.vehicle_age_gt_2_years = 1

        logger.debug(
            "Vehicle age indicators set: 1-2yr=%d, <1yr=%d, >2yr=%d",
            self.vehicle_age_1_2_year,
            self.vehicle_age_lt_1_year,
            self.vehicle_age_gt_2_years,
        )

    def _convert_to_float(self, value: Any) -> Optional[float]:
        """
        Convert form field value to float with null handling.

        Args:
            value: Raw form field value to convert.

        Returns:
            Optional[float]: Converted float value or None if conversion fails.
        """
        if value is None or value == "" or value == "None":
            return None
        try:
            result = float(value)
            logger.debug("Successfully converted '%s' to float: %f", value, result)
            return result
        except (ValueError, TypeError) as e:
            logger.warning("Failed to convert '%s' to float: %s", value, str(e))
            return None

    def _convert_to_int(self, value: Any) -> Optional[int]:
        """
        Convert form field value to integer with null handling.

        Args:
            value: Raw form field value to convert.

        Returns:
            Optional[int]: Converted integer value or None if conversion fails.
        """
        if value is None or value == "" or value == "None":
            return None
        try:
            result = int(
                float(value)
            )  # Convert to float first to handle decimal inputs
            logger.debug("Successfully converted '%s' to int: %d", value, result)
            return result
        except (ValueError, TypeError) as e:
            logger.warning("Failed to convert '%s' to int: %s", value, str(e))
            return None

    def _convert_to_object(self, value: Any) -> Optional[str]:
        """
        Convert form field value to string with null handling.

        Args:
            value: Raw form field value to convert.

        Returns:
            Optional[str]: Converted string value or None if empty.
        """
        if value is None or value == "":
            return None
        result = str(value)
        logger.debug("Successfully converted '%s' to string: %s", value, result)
        return result


@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def index(request: Request) -> HTMLResponse:
    """
    Serve the main application homepage with the prediction form.

    This endpoint renders the main HTML template containing the vehicle
    insurance prediction form interface for user input.

    Args:
        request: FastAPI Request object for template rendering.

    Returns:
        HTMLResponse: Rendered HTML page with the prediction form.
    """
    try:
        logger.info(
            "Serving homepage request from %s",
            request.client.host if request.client else "unknown",
        )

        response = templates.TemplateResponse(
            "index.html", {"request": request, "context": "Rendering"}
        )

        logger.debug("Homepage rendered successfully")
        return response

    except Exception as e:
        logger.error("Template rendering error: %s", str(e), exc_info=True)
        return HTMLResponse(
            content=f"<h1>Template Error</h1><p>Error: {str(e)}</p>", status_code=500
        )


@app.get("/train", tags=["model"])
async def train_route_client() -> Response:
    """
    Trigger the machine learning model training pipeline.

    This endpoint initiates the complete ML training pipeline including
    data ingestion, validation, transformation, model training, evaluation,
    and deployment processes.

    Returns:
        Response: Success message indicating training completion.

    Raises:
        HTTPException: If training pipeline fails with status code 500.
    """
    try:
        logger.info("Starting model training pipeline")

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        logger.info("Model training pipeline completed successfully")
        return Response("Model trained successfully!")

    except Exception as e:
        logger.error("Training pipeline failed: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/", response_class=HTMLResponse, tags=["prediction"])
async def predict_route_client(request: Request) -> HTMLResponse:
    """
    Process insurance prediction request and return results.

    This endpoint handles form submission from the main page, processes
    the vehicle owner data, generates ML model predictions, and returns
    the results rendered in the HTML template.

    Args:
        request: FastAPI Request object containing form data.

    Returns:
        HTMLResponse: Rendered HTML page with prediction results.
    """
    try:
        logger.info(
            "Processing prediction request from %s",
            request.client.host if request.client else "unknown",
        )

        # Extract and validate form data
        data_form = DataForm(request=request)
        await data_form.get_form_data()

        # Create VehicleOwner instance
        logger.debug("Creating VehicleOwner instance from form data")
        owner_data = VehicleOwner(
            age=data_form.age,
            gender=data_form.gender,
            vintage=data_form.vintage,
            region_code=data_form.region_code,
            annual_premium=data_form.annual_premium,
            vehicle_damage=data_form.vehicle_damage,
            driving_license=data_form.driving_license,
            previously_insured=data_form.previously_insured,
            policy_sales_channel=data_form.policy_sales_channel,
            vehicle_age_1_2_year=data_form.vehicle_age_1_2_year,
            vehicle_age_lt_1_year=data_form.vehicle_age_lt_1_year,
            vehicle_age_gt_2_years=data_form.vehicle_age_gt_2_years,
        )

        # Convert to DataFrame for model input
        logger.debug("Converting vehicle owner data to DataFrame")
        owner_data_df = owner_data.vehicle_owner_as_df()

        # Generate prediction
        logger.debug("Initializing model predictor and generating prediction")
        model_predictor = OwnerClassifier()
        prediction = model_predictor.predict(owner_data_df)[0]

        # Interpret prediction result
        status = (
            "Vehicle owner is likely to purchase insurance!"
            if prediction == 1
            else "Vehicle owner is unlikely to purchase insurance."
        )

        logger.info(
            "Prediction completed successfully: result=%d, status='%s'",
            prediction,
            status,
        )

        return templates.TemplateResponse(
            "index.html", {"request": request, "context": status}
        )

    except HTTPException:
        # Re-raise HTTPException to be handled by middleware
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        error_message = f"Prediction Error: {str(e)}"

        try:
            return templates.TemplateResponse(
                "index.html", {"request": request, "context": error_message}
            )
        except Exception as template_error:
            logger.error(
                "Template rendering error: %s", str(template_error), exc_info=True
            )
            return HTMLResponse(
                content=f"<h1>Prediction Error</h1><p>{error_message}</p>",
                status_code=500,
            )


@app.get("/health", tags=["system"])
async def health_check() -> Dict[str, str]:
    """
    Perform application health check and return system status.

    This endpoint provides a simple health check mechanism to verify
    that the FastAPI application is running and responsive.

    Returns:
        Dict[str, str]: Dictionary containing status and message indicating
            application health.
    """
    logger.debug("Health check requested")

    health_status = {
        "status": "Server is running",
        "message": "FastAPI is working correctly",
    }

    logger.debug("Health check completed: %s", health_status)
    return health_status


if __name__ == "__main__":
    logger.info("Starting FastAPI application on %s:%s", APP_HOST, APP_PORT)
    app_run(app, host=APP_HOST, port=APP_PORT)
