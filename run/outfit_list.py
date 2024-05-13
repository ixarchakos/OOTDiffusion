import pandas as pd


def read_sheet(document_id, tab_name):
    full_url = f"https://docs.google.com/spreadsheets/d/{document_id}/gviz/tq?tqx=out:csv&sheet={tab_name}"
    return pd.read_csv(full_url)


def remove_invalid_outfits():
    outfits = read_sheet("1IJNffNBkZVNwZ439llyt6JRDPUH5U_iYbSEUG2bj0v4", "outfits_training_data")
    outfit_set = dict()

    for index, data in outfits.iterrows():
        org_id, out_id, img_url, division = data["org_id"], data["outfit_id"], data["image_url"].split("/")[-1], data["division"]
        # print(org_id, out_id, img_url, division)
        if division != "Tops" and division != "Bottoms":
            continue
        if f"{out_id}_{org_id}" not in outfit_set:
            outfit_set[f"{out_id}_{org_id}"] = [(img_url, division)]
        else:
            outfit_set[f"{out_id}_{org_id}"].append((img_url, division))

    # Delete cases where two Tops appear in the same outfit
    for k, items in outfit_set.copy().items():
        c = 0
        for item in items:
            c += 1 if item[1] == 'Tops' else 0
        if c > 1:
            del outfit_set[k]
    return outfit_set

