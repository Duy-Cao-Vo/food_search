# food_search

## Introduction
`food_search` is a project aimed at crawling and aggregating food-related data from various websites. The project includes pre-built crawlers for specific websites and provides instructions on how to use them to gather and analyze data.

## Websites with existing crawlers
* [shopefood.vn](https://www.shopefood.vn/)
```json
{
    "CityId": 217,
    "ExcludeIds": ",,1603",
    "Query": null,
    "Items": [
        {
            "Id": 902993,
            "Name": "Alo - Cơm Nhà & Cơm Văn Phòng",
            "Address": "85/33 Phạm Viết Chánh, P. 19, Quận Bình Thạnh, TP. HCM",
            "AvgRating": 7.514,
            "AvgRatingText": "7.5",
            "RestaurantStatus": 2,
            "Phone": "Đang cập nhật",
            "PhotoUrl": "https://images.shopefood.vn/res/g91/902993/prof/s640x400/shopefood-upload-api-shopefood-mobile-lll-190509144113.jpg",
            "TotalReviews": 7,
            "TotalFavourites": 0,
            "TotalViews": 0
        }
    ]
}
```
## Download fine-tune model vietnamese Here
Drive: https://drive.google.com/drive/folders/1sQytCi__vDn89fXzktGqxYstWOc_l_87?usp=sharing

## Usage Instructions
1. Clone the repository:
     ```bash
     git clone https://github.com/phamthienvuong98/food_search.git
     ```
2. Install the required libraries:
     ```bash
     pip install -r requirements.txt
     ```
3. Run the crawlers:
     ```bash
     # to crawl new data
     crawl.ipynb
     ```
4. Run the demo:
     ```bash
     python demo.py
     ```
     or open demo.ipynb
