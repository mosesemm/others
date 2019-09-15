const express = require("express");
const router = express.Router();

router.get("/", (req, resp) => {
    resp.json([
        {dateGranted: new Date(), grantedBy: "Mosd Test", grantedTo: "Sipho Test", effectiveRole: ""},
        {dateGranted: new Date(), grantedBy: "Lebo Test", grantedTo: "Lerato Test", effectiveRole: ""},
        {dateGranted: new Date(), grantedBy: "Megail Test", grantedTo: "Thabo Test", effectiveRole: ""}
        ]);
})

module.exports = router;