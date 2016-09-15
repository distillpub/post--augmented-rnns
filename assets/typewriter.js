var tw = {
  register: function(selection) {
    selection.dispatch("tw-ready");
  },
  whenReady: function(selection, callback) {
    callback();
  }

  // whenReady
  // whenResize
  // enterBackstage
  // enterStage
  // enterSpotlight
  // exitSpotlight
  // exitStage
  // exitBackstage

};
