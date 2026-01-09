# SolverX Ontology Schema

이 다이어그램은 `demo_custom_graphrag.py`에서 사용된 온톨로지 스키마와 데이터베이스 구조를 시각화한 것입니다.

```mermaid
classDiagram
    direction LR

    class Company {
        +String name
        +String industry
    }
    class Product {
        +String name
        +String description
    }
    class Technology {
        +String name
        +String function
    }
    class Location {
        +String name
    }
    class Concept {
        +String name
        (e.g., "멀티피직스 모델", "기술")
    }
    class Feature {
        +String description
        (e.g., "10배 빠름")
    }

    %% Relations defined in [Allowed Relations]
    Company "1" --> "1" Location : located_at
    Product "1" --> "1" Company : developed_by
    Product "1" --> "*" Feature : has_feature
    Product "1" --> "1" Concept : is_a
    Product "1" --> "1" Product : outperforms
    Product "1" --> "1" Product : is_similar_to
    
    Technology "1" --> "1" Product : located_at (applied_to)
    Technology "1" --> "*" Feature : has_feature
    Technology "1" --> "1" Concept : is_a

    %% Database Structure Mapping
    note for Company "DB Table: companies"
    note for Product "DB Table: products"
    note for Technology "DB Table: technologies"
```

## 관계 (Relations) 설명

| 관계 (Relation) | 설명 | Domain (주어) | Range (목적어) |
| :--- | :--- | :--- | :--- |
| **developed_by** | 제품을 개발한 회사 | Product | Company |
| **located_at** | 회사 위치 또는 기술이 적용된 제품 | Company, Technology | Location, Product |
| **is_a** | 엔티티의 정의/분류 | Product, Technology | Concept |
| **has_feature** | 주요 기능이나 특징 | Product, Technology | Feature |
| **outperforms** | 경쟁 우위 비교 | Product | Product |
| **is_similar_to** | 제품/기술 간의 유사성 | Product, Technology | Product, Technology |
