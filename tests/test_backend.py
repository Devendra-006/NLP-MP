"""Unit tests for Flask backend API endpoints."""

import io
import json
import pytest
from backend.app import create_app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_root_endpoint(client):
    """Test GET / returns JSON confirmation."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert "message" in data
    assert "News Category Classifier" in data["message"]


def test_health_endpoint(client):
    """Test GET /api/health returns {'status': 'ok'}."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "ok"}


def test_predict_sports(client):
    """Test POST /api/predict with Sports acceptance headline."""
    payload = {
        "text": "Messi scores twice in Champions League final",
        "model": "baseline",
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["label"] == "Sports"
    assert data["confidence"] > 0.80
    assert data["uncertain"] is False
    assert isinstance(data["top_k_probs"], list)
    assert len(data["top_k_probs"]) == 4

    assert "explanation" in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) > 0
    for item in data["explanation"]:
        assert "feature" in item
        assert "contribution" in item
        assert isinstance(item["feature"], str)
        assert isinstance(item["contribution"], (int, float))

    # Verify probability structure
    for item in data["top_k_probs"]:
        assert "label" in item
        assert "probability" in item
        assert 0.0 <= item["probability"] <= 1.0


def test_predict_explanation_endpoint(client):
    """Test POST /api/predict explanation feature across categories."""
    cases = [
        ("NASA Hubble telescope discovers distant supernova explosion", "Sci/Tech"),
        ("Wall Street stock market gains as corporate earnings surge", "Business"),
        ("United Nations security council diplomats sign historic treaty", "World"),
    ]
    for text, expected_label in cases:
        response = client.post("/api/predict", json={"text": text})
        assert response.status_code == 200
        data = response.get_json()
        assert data["label"] == expected_label
        assert "explanation" in data
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0
        features = [item["feature"] for item in data["explanation"]]
        # Stopwords must not appear as features
        assert "the" not in features
        assert "in" not in features


def test_predict_categories(client):
    """Test POST /api/predict across Business, Sci/Tech, and World."""
    cases = [
        ("Wall Street gains as Federal Reserve lowers interest rates", "Business"),
        ("NASA releases new high resolution deep space telescope imagery", "Sci/Tech"),
        ("United Nations representatives attend summit in Geneva", "World"),
    ]
    for text, expected_label in cases:
        response = client.post("/api/predict", json={"text": text})
        assert response.status_code == 200
        data = response.get_json()
        assert data["label"] == expected_label
        assert data["confidence"] > 0.50


def test_predict_empty_text(client):
    """Test POST /api/predict with empty string."""
    response = client.post("/api/predict", json={"text": ""})
    assert response.status_code == 200
    data = response.get_json()
    assert data["label"] == "Unknown"
    assert data["confidence"] == 0.0
    assert data["uncertain"] is True


def test_predict_distilbert_not_available(client):
    """Test POST /api/predict with distilbert model when untrained."""
    response = client.post("/api/predict", json={
        "text": "Breaking news headline",
        "model": "distilbert"
    })
    # Since transformer weights do not exist, should return 400 error
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_batch_predict_csv_json(client):
    """Test POST /api/batch-predict with CSV returning JSON."""
    csv_content = "headline\nMessi scores winning goal\nStock market hits record high\n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test_news.csv"),
        "text_column": "headline",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["total_records"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["predicted_category"] == "Sports"
    assert res["predictions"][1]["predicted_category"] == "Business"


def test_batch_predict_csv_download(client):
    """Test POST /api/batch-predict returning CSV attachment."""
    csv_content = "headline\nMessi scores winning goal\n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test_news.csv"),
        "text_column": "headline",
        "format": "csv",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert "attachment" in response.headers.get("Content-Disposition", "")
    assert b"predicted_category" in response.data


def test_metrics_endpoint(client):
    """Test GET /api/metrics returns real metrics."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.get_json()
    assert "accuracy" in data
    assert "macro_f1" in data
    assert "per_class" in data
    assert "World" in data["per_class"]


def test_confusion_matrix_endpoint(client):
    """Test GET /api/confusion-matrix serves image."""
    response = client.get("/api/confusion-matrix")
    assert response.status_code == 200
    assert response.content_type == "image/png"
    assert len(response.data) > 1000


# ============================================================================
# Task 10: Input Validation & Error Handling Tests
# ============================================================================

def test_predict_whitespace_text(client):
    """Test POST /api/predict with whitespace-only text."""
    response = client.post("/api/predict", json={"text": "     \t\n  "})
    assert response.status_code == 200
    data = response.get_json()
    assert data["label"] == "Unknown"
    assert data["confidence"] == 0.0
    assert data["uncertain"] is True
    assert "warning" in data


def test_predict_short_text(client):
    """Test POST /api/predict with extremely short text triggers uncertainty and warning."""
    response = client.post("/api/predict", json={"text": "Hi"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["uncertain"] is True
    assert "warning" in data
    assert "too short" in data["warning"]


def test_predict_extremely_long_text(client):
    """Test POST /api/predict with text > 50,000 characters returns 400 error."""
    long_text = "Breaking news headline: " + ("word " * 11000)  # > 55,000 chars
    response = client.post("/api/predict", json={"text": long_text})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "too long" in data["error"]


def test_predict_missing_text_field(client):
    """Test POST /api/predict with missing or null text field returns 400."""
    # Missing field
    response = client.post("/api/predict", json={"model": "baseline"})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "news text" in data["error"]

    # Null field
    response2 = client.post("/api/predict", json={"text": None})
    assert response2.status_code == 400
    data2 = response2.get_json()
    assert "error" in data2


def test_predict_invalid_model_selection(client):
    """Test POST /api/predict with non-existent model name returns 400."""
    response = client.post("/api/predict", json={
        "text": "Valid headline about soccer championship",
        "model": "non_existent_gpt9",
    })
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Invalid model selection" in data["error"]


def test_predict_non_json_request(client):
    """Test POST /api/predict with plain text content-type returns 400."""
    response = client.post(
        "/api/predict",
        data="just plain text without json",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_batch_no_file(client):
    """Test POST /api/batch-predict with no file returns 400."""
    response = client.post("/api/batch-predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "No file" in data["error"]


def test_batch_invalid_file_type(client):
    """Test POST /api/batch-predict with non-CSV file returns 400."""
    data = {
        "file": (io.BytesIO(b"some text"), "data.txt"),
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Invalid file type" in data["error"]


def test_batch_empty_csv(client):
    """Test POST /api/batch-predict with empty CSV returns 400."""
    data = {
        "file": (io.BytesIO(b""), "empty.csv"),
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_batch_missing_column(client):
    """Test POST /api/batch-predict with requested column not present returns 400."""
    csv_content = "headline\nStock market news\n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "test.csv"),
        "text_column": "non_existent_column_name",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Please select a valid text column" in data["error"]


def test_batch_empty_rows_only(client):
    """Test POST /api/batch-predict with text column containing only whitespace returns 400."""
    csv_content = "headline\n   \n\n  \n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "empty_rows.csv"),
        "text_column": "headline",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_batch_partial_empty_rows(client):
    """Test POST /api/batch-predict with some blank rows processes without crashing."""
    csv_content = "headline\nMessi scores twice in Champions League\n\nWall Street gains on earnings\n"
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "partial.csv"),
        "text_column": "headline",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["total_records"] == 3
    # First row is sports
    assert res["predictions"][0]["predicted_category"] == "Sports"
    # Second (empty) row is Unknown or uncertain
    assert res["predictions"][1]["predicted_category"] == "Unknown"
    assert res["predictions"][1]["uncertain"] is True
    # Third row is business
    assert res["predictions"][2]["predicted_category"] == "Business"


def test_batch_exceeds_max_rows(client):
    """Test POST /api/batch-predict with CSV exceeding MAX_BATCH_ROWS returns 400."""
    lines = ["headline"] + ["Sample news headline"] * 10005
    csv_content = "\n".join(lines)
    data = {
        "file": (io.BytesIO(csv_content.encode("utf-8")), "large.csv"),
        "text_column": "headline",
    }
    response = client.post("/api/batch-predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res = response.get_json()
    assert "exceeds the maximum limit" in res["error"]

