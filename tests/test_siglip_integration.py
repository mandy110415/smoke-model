from PIL import Image
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

import audit_service


def test_verify_passes_siglip_evidence_to_vlm(monkeypatch):
    image = Image.new('RGB', (100, 100), 'white')
    evidence = {
        'available': True,
        'fused_scores': {'smoke': 0.7, 'cloud': 0.3},
        'smoke_cloud_margin': 0.4,
    }

    monkeypatch.setattr(audit_service, 'analyze_region', lambda **kwargs: evidence)

    captured = {}

    def fake_verify_region(**kwargs):
        captured.update(kwargs)
        return {
            'label': 'SMOKE',
            'siglip_evidence': kwargs['siglip_evidence'],
        }

    monkeypatch.setattr(audit_service, 'verify_region', fake_verify_region)

    result = audit_service.verify(
        image=image,
        box=[20, 20, 40, 40],
        source='existing_annotation',
    )

    assert result['label'] == 'SMOKE'
    assert captured['siglip_evidence'] == evidence
    assert captured['context_image'].size[0] >= captured['crop_image'].size[0]
    assert captured['context_image'].size[1] >= captured['crop_image'].size[1]
