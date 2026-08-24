"""
HealthSense - Interactive API Documentation (OpenAPI 3.0 & Swagger UI)
Serves Swagger UI at /docs and OpenAPI 3.0 specification at /api/openapi.json.
"""

from flask import Blueprint, jsonify, render_template_string

docs_bp = Blueprint("docs_bp", __name__)

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "HealthSense ECG AI & Telemetry API",
        "description": "Production-ready REST API for AI-assisted ECG arrhythmia screening and real-time vital telemetry.",
        "version": "2.0.0"
    },
    "servers": [
        {"url": "/", "description": "Local Development Server"}
    ],
    "paths": {
        "/api/health": {
            "get": {
                "summary": "System Health Check",
                "description": "Returns operational status, model file name, and active decision threshold.",
                "responses": {
                    "200": {
                        "description": "Service is healthy",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "healthy",
                                    "service": "HealthSense API",
                                    "model": "ecg_model_optimized.keras",
                                    "threshold": 0.30
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ecg/model-info": {
            "get": {
                "summary": "Model Metadata & Validated Benchmark Metrics",
                "description": "Returns safe metadata and previously validated test-set metrics (79.02% acc, 55.03% prec, 57.63% rec, 85.58% spec, 0.5630 F1, 0.8056 ROC-AUC).",
                "responses": {
                    "200": {
                        "description": "Model metadata retrieved successfully",
                        "content": {
                            "application/json": {
                                "example": {
                                    "model": "Optimized 1D-CNN",
                                    "input_length": 256,
                                    "threshold": 0.30,
                                    "accuracy": 0.7902,
                                    "precision": 0.5503,
                                    "recall": 0.5763,
                                    "specificity": 0.8558,
                                    "f1_score": 0.5630,
                                    "roc_auc": 0.8056,
                                    "preprocessing": "Per-beat Z-score normalization (mean 0, variance 1)",
                                    "test_dataset": "58,448 beats across 26 unseen patient records",
                                    "status": "LOADED"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ecg/predict": {
            "post": {
                "summary": "Predict Arrhythmia on 256 ECG Samples",
                "description": "Accepts exactly 256 raw or preprocessed ECG amplitude samples and returns binary classification, abnormal probability, confidence, and validated 0.30 decision threshold.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["ecg"],
                                "properties": {
                                    "ecg": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 256,
                                        "maxItems": 256,
                                        "description": "Exactly 256 numerical ECG voltage/amplitude points."
                                    }
                                }
                            },
                            "example": {
                                "ecg": [0.05, 0.08, 0.12, 0.20, 0.85, 1.20, 0.15, -0.22, 0.01] + [0.0] * 247
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Successful prediction",
                        "content": {
                            "application/json": {
                                "examples": {
                                    "Normal Beat": {
                                        "value": {
                                            "prediction": "NORMAL",
                                            "abnormal_probability": 0.008877,
                                            "confidence": 0.991123,
                                            "threshold": 0.30,
                                            "class_id": 0,
                                            "status": "SUCCESS"
                                        }
                                    },
                                    "Abnormal Beat": {
                                        "value": {
                                            "prediction": "ABNORMAL",
                                            "abnormal_probability": 0.991265,
                                            "confidence": 0.991265,
                                            "threshold": 0.30,
                                            "class_id": 1,
                                            "status": "SUCCESS"
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Validation error (invalid length, NaN, Inf, non-numeric)",
                        "content": {
                            "application/json": {
                                "example": {
                                    "error": "Invalid ECG window length: 255 samples. Expected exactly 256 samples.",
                                    "error_code": "VALIDATION_ERROR",
                                    "status": "ERROR"
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/ecg/sample-beats": {
            "get": {
                "summary": "Get Verified Sample Beats",
                "description": "Returns verified Normal (index 0) and Abnormal/PVC (index 6077) beats for testing.",
                "responses": {
                    "200": {
                        "description": "Sample beats returned"
                    }
                }
            }
        },
        "/api/auth/register": {
            "post": {
                "summary": "Register New Patient Account",
                "description": "Creates a new user account with hashed password and unique auto-generated Patient ID.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "example": {
                                "full_name": "John Doe",
                                "email": "john@example.com",
                                "password": "secure-password",
                                "confirm_password": "secure-password",
                                "phone": "+91 98765 43210",
                                "age": 30,
                                "gender": "Male"
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "User created successfully"
                    },
                    "400": {
                        "description": "Validation error or duplicate email"
                    }
                }
            }
        },
        "/api/auth/login": {
            "post": {
                "summary": "Authenticate User & Start Session",
                "description": "Validates email and password, initializes cryptographically signed Flask session.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "example": {
                                "email": "john@example.com",
                                "password": "secure-password"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Login successful"
                    },
                    "401": {
                        "description": "Invalid credentials"
                    }
                }
            }
        },
        "/api/auth/logout": {
            "post": {
                "summary": "Logout & Clear Session",
                "description": "Clears the active authenticated session.",
                "responses": {
                    "200": {
                        "description": "Logged out successfully"
                    }
                }
            }
        },
        "/api/auth/me": {
            "get": {
                "summary": "Get Current Session User",
                "description": "Returns currently logged-in user profile or unauthenticated state.",
                "responses": {
                    "200": {
                        "description": "User session state returned"
                    }
                }
            }
        }
    }
}

SWAGGER_UI_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HealthSense API Documentation — Swagger UI</title>
  <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@5/favicon-32x32.png" sizes="32x32" />
  <style>
    html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
    *, *:before, *:after { box-sizing: inherit; }
    body { margin: 0; background: #0b1120; color: #f8fafc; font-family: sans-serif; }
    .topbar { display: none !important; }
    .swagger-ui { color: #f8fafc !important; }
    .swagger-ui .info .title { color: #38bdf8 !important; }
    .swagger-ui .info p, .swagger-ui .info li { color: #cbd5e1 !important; }
    .swagger-ui .scheme-container { background: #0f172a !important; box-shadow: none !important; }
    .swagger-ui .opblock { border-radius: 8px !important; border: 1px solid #1e293b !important; }
    .swagger-ui .opblock.opblock-post { background: rgba(14, 165, 233, 0.08) !important; border-color: #0ea5e9 !important; }
    .swagger-ui .opblock.opblock-get { background: rgba(16, 185, 129, 0.08) !important; border-color: #10b981 !important; }
    .swagger-ui .opblock-summary-description { color: #94a3b8 !important; }
    .swagger-ui section.models { border: 1px solid #1e293b !important; border-radius: 8px !important; background: #0f172a !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
  <script>
    window.onload = function() {
      SwaggerUIBundle({
        url: "/api/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>
"""


@docs_bp.route("/docs", methods=["GET"])
def swagger_ui():
    """Renders interactive Swagger UI for API documentation."""
    return render_template_string(SWAGGER_UI_TEMPLATE)


@docs_bp.route("/api/openapi.json", methods=["GET"])
def openapi_json():
    """Returns the OpenAPI 3.0 specification JSON."""
    return jsonify(OPENAPI_SPEC)
